use cubecl::{
    zspace::Shape,
    {VectorizationError, prelude::*},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    components::batch::gemm_plane_parallel::{MatmulOperandLayouts, OperandLayout},
    definition::cube_mapping_launch,
    definition::{MatmulElems, MatmulProblem, MatmulSetupError, MatmulVectorSizes},
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

    // Tentative problem to classify the kind. Every layout combination
    // is supported — including (Col, Row), which routes through the
    // double-staged path. Only vec operands need contiguity fix-ups.
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

    let kind = MatmulOperandLayouts::from_problem(&kind_problem)?;

    // Vec operands need K-contiguous storage (stride along K = 1). Mat
    // operands keep their natural layout — the kernel selects the right
    // traversal (Direct / StageRhs / StageLhs / DoubleStaged) from the
    // resulting `MatmulKind`.
    if matches!(kind.lhs, OperandLayout::Vector) && kind_problem.lhs_strides[rank - 1] != 1 {
        lhs = lhs.into_contiguous(client)?;
    }
    if matches!(kind.rhs, OperandLayout::Vector) && kind_problem.rhs_strides[rank - 1] != 1 {
        rhs = rhs.into_contiguous(client)?;
    }
    let final_lhs_layout = lhs_layout;
    let final_rhs_layout = rhs_layout;

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

    // The staged-tile paths are CPU-only for now (the kernel writes one
    // `vector_size`-wide chunk per mn_id, which doesn't fully cover the
    // output when plane_dim > 1).
    if device_settings.plane_dim > 1 && expand_info.blueprint.kind.k_traversal().is_staged() {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "On GPU, staged plane-parallel paths are not supported yet (kind={:?})",
            expand_info.blueprint.kind
        ))));
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
