//! No Stage VecMat matmul kernel implementation
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::zspace::Shape;
use cubecl::{VectorizationError, prelude::*};
use cubek_std::{InputBinding, MatrixLayout};

use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError};
use crate::definition::{MatmulVectorSizes, cube_mapping_launch};

use crate::launch::InputArg;
use crate::launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs};
use crate::routines::nostage_vecmat::{NoStageVecMatRoutine, NoStageVecMatStrategy};
use crate::routines::{BlueprintStrategy, Routine as _};

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: InputBinding<R>,
    rhs: InputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = rhs.shape().len();

    // Rhs is assumed row major for now
    let rhs_layout = matrix_batch_layout(&rhs.data().strides, rhs.scheme());
    let rhs = if !matches!(rhs_layout, MatrixBatchLayout::Contiguous) {
        rhs.into_contiguous(client)?
    } else {
        rhs
    };

    let m = lhs.shape().to_vec()[rank - 2];
    let n = rhs.shape().to_vec()[rank - 1];
    let k = lhs.shape().to_vec()[rank - 1];

    if m != 1 {
        return Err(MatmulSetupError::InvalidConfig(Box::new("m must equal 1")));
    }

    let rhs_shape = rhs.shape();
    // let out_shape = &out.shape;

    // Assumes row-major
    let lhs_supported_vector_sizes = client.io_optimized_vector_sizes(dtypes.lhs_global.size());
    let divisible =
        *lhs.data().shape.last().unwrap() / client.properties().hardware.plane_size_max as usize;
    let mut lhs_vector_size = lhs_supported_vector_sizes
        .filter(|&vector_size| divisible.is_multiple_of(vector_size))
        .max()
        .ok_or(VectorizationError::NoValidVectorization)?;
    let rhs_supported_vector_sizes = client.io_optimized_vector_sizes(dtypes.rhs_global.size());
    let divisible =
        *rhs.data().shape.last().unwrap() / client.properties().hardware.plane_size_max as usize;
    let mut rhs_vector_size = rhs_supported_vector_sizes
        .filter(|&vector_size| divisible.is_multiple_of(vector_size))
        .max()
        .ok_or(VectorizationError::NoValidVectorization)?;

    if let InputBinding::Quantized { scheme, .. } = lhs {
        lhs_vector_size *= scheme.num_quants();
    }
    if let InputBinding::Quantized { scheme, .. } = rhs {
        rhs_vector_size *= scheme.num_quants();
    }

    if lhs_vector_size != rhs_vector_size {
        return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
            "Lhs vector size {:?} must equal rhs vector size {:?}",
            lhs_vector_size, rhs_vector_size
        ))));
    }

    let vector_sizes = MatmulVectorSizes {
        lhs: rhs_vector_size,
        rhs: rhs_vector_size,
        out: rhs_vector_size,
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

    let device_settings = NoStageVecMatRoutine::device_settings(client, vector_sizes);
    let expand_info = NoStageVecMatRoutine::expand_blueprint(
        &problem,
        &device_settings,
        &BlueprintStrategy::Inferred(NoStageVecMatStrategy {
            target_num_planes: 8,
        }),
    )?;
    let launch_info = NoStageVecMatRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<NoStageVecMatRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<NoStageVecMatRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );

    NoStageVecMatRoutine::launch::<TensorArgs, R>(
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
