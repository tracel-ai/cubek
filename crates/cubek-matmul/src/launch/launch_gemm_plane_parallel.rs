use cubecl::{
    zspace::Shape,
    {VectorizationError, prelude::*},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    components::batch::gemm_plane_parallel::MatmulKind,
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    definition::{MatmulVectorSizes, cube_mapping_launch},
};

use crate::{
    launch::InputArg,
    launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs},
    routines::gemm_plane_parallel::GemmPlaneParallelRoutine,
    routines::{BlueprintStrategy, Routine as _},
};

fn vector_size_for<R: Runtime>(
    client: &ComputeClient<R>,
    binding: &InputBinding<R>,
    default_size: usize,
    plane_size: usize,
    dim: usize,
) -> Result<usize, VectorizationError> {
    let (size, num_quants) = if let InputBinding::Quantized { scheme, .. } = binding {
        (scheme.size_bits_stored() / 8, scheme.num_quants())
    } else {
        (default_size, 1)
    };
    client
        .io_optimized_vector_sizes(size)
        .filter(|&v| dim.is_multiple_of(plane_size * v * num_quants))
        .max()
        .ok_or(VectorizationError::NoValidVectorization)
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    mut lhs: InputBinding<R>,
    mut rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), GemmPlaneParallelRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = rhs.shape().len();

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();

    let m = lhs_shape.to_vec()[rank - 2];
    let n = rhs_shape.to_vec()[rank - 1];
    let k = lhs_shape.to_vec()[rank - 1];

    let plane_size = client.properties().hardware.plane_size_max as usize;

    if !k.is_multiple_of(plane_size) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Dimension k={} must be a multiple of plane size {}",
            k, plane_size
        ))));
    }

    let lhs_vector_size = vector_size_for(client, &lhs, dtypes.lhs_global.size(), plane_size, k)?;
    let rhs_vector_size = vector_size_for(client, &rhs, dtypes.rhs_global.size(), plane_size, k)?;

    let shared_vector_size = lhs_vector_size.min(rhs_vector_size);

    let vector_sizes = MatmulVectorSizes {
        lhs: shared_vector_size,
        rhs: shared_vector_size,
        out: 1,
    };

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let lhs_batches: Shape = lhs.shape().to_vec()[..rank - 2].into();
    let rhs_batches: Shape = rhs.shape().to_vec()[..rank - 2].into();

    let lhs_layout =
        MatrixLayout::from_shape_and_strides(lhs_shape, &lhs.data().strides, lhs.scheme())?;
    let rhs_layout =
        MatrixLayout::from_shape_and_strides(rhs_shape, &rhs.data().strides, rhs.scheme())?;

    // Tentative problem just to classify the kind. For the matmat case we
    // overwrite the layouts to RowMajor/ColMajor below (after forcing
    // contiguity), since that's the only matmat layout this kernel supports.
    let kind_problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches.clone(),
        rhs_batches.clone(),
        lhs_layout,
        rhs_layout,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let kind = MatmulKind::from_problem(&kind_problem)?;

    // Force the operand stride layout this kind needs:
    // - matmat: lhs RowMajor (K-stride = 1), rhs ColMajor (K-stride = 1).
    // - vec-mat: vec (lhs) contiguous along K.
    // - mat-vec: vec (rhs) contiguous along K.
    let (final_lhs_layout, final_rhs_layout) = match kind {
        MatmulKind::MatRowMatCol => {
            if kind_problem.lhs_strides[rank - 1] != 1 {
                lhs = lhs.into_contiguous(client)?;
            }
            if kind_problem.rhs_strides[rank - 2] != 1 {
                rhs = rhs.into_contiguous(client)?;
            }
            (MatrixLayout::RowMajor, MatrixLayout::ColMajor)
        }
        MatmulKind::VecMatCol | MatmulKind::VecMatRow => {
            if kind_problem.lhs_strides[rank - 1] != 1 {
                lhs = lhs.into_contiguous(client)?;
            }
            (lhs_layout, rhs_layout)
        }
        MatmulKind::MatRowVec | MatmulKind::MatColVec => {
            if kind_problem.rhs_strides[rank - 1] != 1 {
                rhs = rhs.into_contiguous(client)?;
            }
            (lhs_layout, rhs_layout)
        }
    };

    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        final_lhs_layout,
        final_rhs_layout,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let device_settings = GemmPlaneParallelRoutine::device_settings(client, vector_sizes);
    let expand_info =
        GemmPlaneParallelRoutine::expand_blueprint(&problem, &device_settings, strategy)?;

    // The staged-tile kinds are CPU-only for now (kernel writes one
    // `vector_size`-wide chunk per mn_id, which doesn't fully cover the
    // output when plane_dim > 1).
    if device_settings.plane_dim > 1 {
        if matches!(expand_info.blueprint.kind, MatmulKind::MatColVec) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "On GPU, MatVec plane parallel only supports row-major lhs for now",
            )));
        } else if matches!(expand_info.blueprint.kind, MatmulKind::VecMatRow) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "On GPU, VecMat plane parallel only supports col-major rhs for now",
            )));
        }
    }

    let launch_info = GemmPlaneParallelRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<GemmPlaneParallelRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<GemmPlaneParallelRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &launch_info.vector_sizes,
        dtypes,
    );

    GemmPlaneParallelRoutine::launch::<TensorArgs, R>(
        client,
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve(),
        launch_info.address_type,
        input,
        output,
        (),
        cube_mapping_launch(&launch_info.cube_count_plan),
        launch_info.blueprint,
        dtypes,
        &launch_info.vector_sizes,
    )
}
