//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::tensor_vector_size_parallel;

use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError};
use crate::definition::{MatmulVectorSizes, cube_mapping_launch};

use crate::launch::InputArg;
use crate::launch::handle::MatmulInputBinding;
use crate::launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs};
use crate::routines::naive::NaiveRoutine;
use crate::routines::{BlueprintStrategy, Routine as _};

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: MatmulInputBinding<R>,
    rhs: MatmulInputBinding<R>,
    out: TensorBinding<R>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = lhs.shape().len();
    let dim1 = rank - 1;
    let dim2 = rank - 2;

    let lhs_layout = matrix_batch_layout(&lhs.data().strides, lhs.scheme());
    let rhs_layout = matrix_batch_layout(&rhs.data().strides, rhs.scheme());

    let lhs = if !matches!(lhs_layout, MatrixBatchLayout::Contiguous) {
        lhs.into_contiguous(client)?
    } else {
        lhs
    };

    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let correct_rhs_layout = |mut rhs: MatmulInputBinding<R>| {
        rhs.swap_dims(dim1, dim2);
        let mut rhs = rhs.into_contiguous(client)?;

        rhs.swap_dims(dim1, dim2);

        let returned: Result<MatmulInputBinding<R>, LaunchError> = Ok(rhs);
        returned
    };

    let rhs = match rhs_layout {
        MatrixBatchLayout::Contiguous => correct_rhs_layout(rhs)?,
        MatrixBatchLayout::MildlyPermuted {
            transposed,
            batch_swap,
        } => {
            if transposed && !batch_swap {
                rhs
            } else {
                correct_rhs_layout(rhs)?
            }
        }
        MatrixBatchLayout::HighlyPermuted => correct_rhs_layout(rhs)?,
    };

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let out_shape = &out.shape;

    let lhs_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtypes.lhs_global.size()),
        &lhs.data().shape,
        &lhs.data().strides,
        rank - 1,
    );
    let rhs_vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtypes.rhs_global.size()),
        &rhs.data().shape,
        &rhs.data().strides,
        rank - 2,
    );
    let vector_sizes = MatmulVectorSizes {
        lhs: lhs_vector_size,
        rhs: rhs_vector_size,
        out: 1,
    };

    let mut view_vector_sizes = vector_sizes;

    if let MatmulInputBinding::Quantized { scheme, .. } = lhs {
        view_vector_sizes.lhs *= scheme.num_quants();
    }
    if let MatmulInputBinding::Quantized { scheme, .. } = rhs {
        view_vector_sizes.rhs *= scheme.num_quants();
    }

    let address_type = lhs
        .required_address_type()
        .max(rhs.required_address_type())
        .max(out.required_address_type(dtypes.acc_global.size()));

    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.into(),
        rhs_shape.into(),
        out_shape.into(),
        lhs.data().strides.clone(),
        rhs.data().strides.clone(),
        out.strides.clone(),
        dtypes.as_global_elems(),
        address_type,
        lhs.scheme(),
        rhs.scheme(),
    )?;

    let device_settings = NaiveRoutine::device_settings(client, view_vector_sizes);
    let expand_info = NaiveRoutine::expand_blueprint(
        &problem,
        &device_settings,
        &BlueprintStrategy::Inferred(().into()),
    )?;
    let launch_info = NaiveRoutine::prepare(&problem, &device_settings, expand_info)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<NaiveRoutine>>::create(
        lhs,
        rhs,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<NaiveRoutine>>::create(
        out,
        &launch_info.blueprint,
        &problem,
        &vector_sizes,
        dtypes,
    );

    NaiveRoutine::launch::<TensorArgs, R>(
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
