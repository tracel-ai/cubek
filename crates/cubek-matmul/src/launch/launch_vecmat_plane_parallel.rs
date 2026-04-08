use cubecl::{
    zspace::Shape,
    {VectorizationError, prelude::*},
};
use cubek_std::{InputBinding, MatrixLayout};

use crate::{
    components::batch::gemv_plane_parallel::GemvKind,
    definition::{MatmulElems, MatmulProblem, MatmulSetupError},
    definition::{MatmulVectorSizes, cube_mapping_launch},
};

use crate::{
    launch::InputArg,
    launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs},
    routines::vecmat_plane_parallel::GemvPlaneParallelRoutine,
    routines::{BlueprintStrategy, Routine as _},
};

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), GemvPlaneParallelRoutine>,
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

    let lhs_vector_size = client
        .io_optimized_vector_sizes(dtypes.lhs_global.size())
        .map(|v| {
            if let InputBinding::Quantized { scheme, .. } = lhs {
                v * scheme.num_quants()
            } else {
                v
            }
        })
        .filter(|&v| k.is_multiple_of(plane_size * v))
        .max()
        .ok_or(VectorizationError::NoValidVectorization)?;

    let rhs_vector_size = client
        .io_optimized_vector_sizes(dtypes.rhs_global.size())
        .map(|v| {
            if let InputBinding::Quantized { scheme, .. } = rhs {
                v * scheme.num_quants()
            } else {
                v
            }
        })
        .filter(|&v| k.is_multiple_of(plane_size * v))
        .max()
        .ok_or(VectorizationError::NoValidVectorization)?;

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

    let problem = MatmulProblem::from_parameters(
        m,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::from_shape_and_strides(lhs_shape, &lhs.data().strides, lhs.scheme())?,
        MatrixLayout::from_shape_and_strides(rhs_shape, &rhs.data().strides, rhs.scheme())?,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let device_settings = GemvPlaneParallelRoutine::device_settings(client, vector_sizes);
    let expand_info =
        GemvPlaneParallelRoutine::expand_blueprint(&problem, &device_settings, strategy)?;

    if device_settings.plane_dim > 1 {
        if matches!(expand_info.blueprint.kind, GemvKind::MatVecColMajor) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "On GPU, MatVec plane parallel only supports row major lhs for now",
            )));
        } else if matches!(expand_info.blueprint.kind, GemvKind::VecMatRowMajor) {
            return Err(MatmulSetupError::InvalidConfig(Box::new(
                "On GPU, Vecmat plane parallel only supports col major rhs for now",
            )));
        }
    }

    let launch_info = GemvPlaneParallelRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<GemvPlaneParallelRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<GemvPlaneParallelRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );

    GemvPlaneParallelRoutine::launch::<TensorArgs, R>(
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
