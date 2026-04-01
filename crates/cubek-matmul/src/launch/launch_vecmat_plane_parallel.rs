use cubecl::zspace::Shape;
use cubecl::{VectorizationError, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};

use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError};
use crate::definition::{MatmulVectorSizes, cube_mapping_launch};

use crate::launch::InputArg;
use crate::launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs};
use crate::routines::vecmat_plane_parallel::VecMatPlaneParallelRoutine;
use crate::routines::{BlueprintStrategy, Routine as _};

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    strategy: &BlueprintStrategy<(), VecMatPlaneParallelRoutine>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = rhs.shape().len();

    let m = lhs.shape().to_vec()[rank - 2];
    let n = rhs.shape().to_vec()[rank - 1];
    let k = lhs.shape().to_vec()[rank - 1];

    if m != 1 {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "m must equal 1 to qualify as a vecmat problem",
        )));
    }

    let rhs_shape = rhs.shape();

    let plane_size = client.properties().hardware.plane_size_max as usize;

    if !k.is_multiple_of(plane_size) {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Lhs dimension k={} must be a multiple of plane size {}",
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

    // Assumes rhs is col major
    let rhs_vector_size = client
        .io_optimized_vector_sizes(dtypes.rhs_global.size())
        .map(|v| {
            if let InputBinding::Quantized { scheme, .. } = rhs {
                v * scheme.num_quants()
            } else {
                v
            }
        })
        .filter(|&v| n.is_multiple_of(plane_size * v))
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
        1,
        n,
        k,
        lhs_batches,
        rhs_batches,
        MatrixLayout::RowMajor,
        MatrixLayout::from_shape_and_strides(rhs_shape, &rhs.data().strides, rhs.scheme())?,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    if problem.rhs_layout != MatrixLayout::ColMajor {
        return Err(MatmulSetupError::InvalidConfig(Box::new(
            "Vecmat plane parallel only supports col major rhs for now",
        )));
    }

    let device_settings = VecMatPlaneParallelRoutine::device_settings(client, vector_sizes);
    let expand_info =
        VecMatPlaneParallelRoutine::expand_blueprint(&problem, &device_settings, strategy)?;
    let launch_info = VecMatPlaneParallelRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<VecMatPlaneParallelRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );
    let output =
        <OutputArg<TensorArgs> as ConcreteOutputFactory<VecMatPlaneParallelRoutine>>::create(
            out,
            &launch_info.blueprint,
            &problem,
            &vector_sizes,
            dtypes,
        );

    VecMatPlaneParallelRoutine::launch::<TensorArgs, R>(
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
