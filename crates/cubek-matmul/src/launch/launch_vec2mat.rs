//! Vec2Mat matmul kernel implementation
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::tensor_vector_size_parallel;
use cubecl::zspace::shape;
use cubecl::{VectorizationError, prelude::*};
use cubek_std::MatrixLayout;

use crate::definition::MatmulVectorSizes;
use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError};

use crate::launch::InputArg;
use crate::launch::handle::MatmulInputBinding;
use crate::launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs};
use crate::routines::naive::NaiveRoutine;
use crate::routines::vec2mat::Vec2MatRoutine;
use crate::routines::{BlueprintStrategy, Routine as _};

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: MatmulInputBinding<R>,
    rhs: MatmulInputBinding<R>,
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
    let out_shape = &out.shape;

    let mut lhs_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtypes.lhs_global.size()),
        &lhs.data().shape,
        &lhs.data().strides,
        rank - 1,
    );
    let mut rhs_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtypes.rhs_global.size()),
        &rhs.data().shape,
        &rhs.data().strides,
        rank - 1,
    );

    if let MatmulInputBinding::Quantized { scheme, .. } = lhs {
        lhs_vector_size *= scheme.num_quants();
    }
    if let MatmulInputBinding::Quantized { scheme, .. } = rhs {
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

    let problem = MatmulProblem::from_parameters(
        1,
        n,
        k,
        shape![1],
        shape![1],
        MatrixLayout::RowMajor,
        MatrixLayout::from_shape_and_strides(&rhs_shape, &rhs.data().strides, rhs.scheme())?,
        MatrixLayout::RowMajor,
        lhs.scheme(),
        rhs.scheme(),
        dtypes.as_global_elems(),
        address_type,
    );

    let device_settings = Vec2MatRoutine::device_settings(client, vector_sizes);
    let expand_info = Vec2MatRoutine::expand_blueprint(
        &problem,
        &device_settings,
        &BlueprintStrategy::Inferred(().into()),
    )?;
    let launch_info = Vec2MatRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<Vec2MatRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<Vec2MatRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );

    Vec2MatRoutine::launch::<TensorArgs, R>(
        client,
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve(),
        launch_info.address_type,
        input,
        output,
        (),
        launch_info.cube_count_plan.as_args(),
        launch_info.blueprint,
        dtypes,
        &launch_info.vector_sizes,
    )
}
