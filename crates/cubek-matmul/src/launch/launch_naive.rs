//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::tensor_line_size_parallel;

use crate::definition::MatmulLineSizes;
use crate::definition::{MatmulElems, MatmulProblem, MatmulSetupError};

use crate::launch::InputArg;
use crate::launch::handle::{MatmulInputHandle, MatmulInputHandleRef};
use crate::launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs};
use crate::routines::naive::NaiveRoutine;
use crate::routines::{BlueprintStrategy, Routine as _};

/// Matrix multiplication using memory coalescing algorithm with custom cube dimensions
#[allow(clippy::result_large_err)]
pub fn launch<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: MatmulInputHandle<R>,
    rhs: MatmulInputHandle<R>,
    out: &TensorHandleRef<'_, R>,
    dtypes: MatmulElems,
) -> Result<(), MatmulSetupError> {
    launch_ref(client, &lhs.as_ref(), &rhs.as_ref(), out, &dtypes)
}

#[allow(clippy::result_large_err)]
pub fn launch_ref<R: Runtime>(
    client: &ComputeClient<R>,
    lhs: &MatmulInputHandleRef<'_, R>,
    rhs: &MatmulInputHandleRef<'_, R>,
    out: &TensorHandleRef<'_, R>,
    dtypes: &MatmulElems,
) -> Result<(), MatmulSetupError> {
    let rank = lhs.shape().len();
    let dim1 = rank - 1;
    let dim2 = rank - 2;

    let lhs_layout = matrix_batch_layout(lhs.data().strides, lhs.scheme());
    let rhs_layout = matrix_batch_layout(rhs.data().strides, rhs.scheme());

    let lhs = if !matches!(lhs_layout, MatrixBatchLayout::Contiguous) {
        lhs.into_contiguous(client)?
    } else {
        MatmulInputHandle::from_ref(lhs)
    };
    let lhs = lhs.as_ref();
    let rhs = MatmulInputHandle::from_ref(rhs);

    // we swap the dimensions to achieve memory-coalescing:
    // consecutive elements of a column in the original rhs tensor will now be stored
    // consecutively in memory, which allows to fetch them with fewer memory instructions
    let correct_rhs_layout = |mut rhs: MatmulInputHandle<R>| {
        rhs.swap_dims(dim1, dim2);
        let mut rhs = rhs.as_ref().into_contiguous(client)?;

        rhs.swap_dims(dim1, dim2);

        let returned: Result<MatmulInputHandle<R>, LaunchError> = Ok(rhs);
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
    let rhs = rhs.as_ref();

    let lhs_shape = lhs.shape();
    let rhs_shape = rhs.shape();
    let out_shape = out.shape;

    let lhs_line_size = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtypes.lhs_global),
        lhs.data().shape,
        lhs.data().strides,
        rank - 1,
    );
    let rhs_line_size = tensor_line_size_parallel(
        client.io_optimized_line_sizes(&dtypes.rhs_global),
        rhs.data().shape,
        rhs.data().strides,
        rank - 2,
    );
    let line_sizes = MatmulLineSizes {
        lhs: lhs_line_size,
        rhs: rhs_line_size,
        out: 1,
    };

    let problem = MatmulProblem::from_shapes_and_strides(
        lhs_shape.to_vec(),
        rhs_shape.to_vec(),
        out_shape.to_vec(),
        lhs.data().strides.to_vec(),
        rhs.data().strides.to_vec(),
        out.strides.to_vec(),
        dtypes.as_global_elems(),
        lhs.scheme(),
        rhs.scheme(),
    );

    let device_settings = NaiveRoutine::device_settings(client, line_sizes);
    let launch_info = NaiveRoutine::prepare(
        &problem,
        &device_settings,
        &BlueprintStrategy::Inferred(().into()),
    )?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<NaiveRoutine>>::create(
        client,
        &lhs,
        &rhs,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<NaiveRoutine>>::create(
        client,
        out,
        &launch_info.blueprint,
        &problem,
        &line_sizes,
        dtypes,
    );

    NaiveRoutine::launch::<TensorArgs, R>(
        client,
        launch_info.cube_dim,
        launch_info.cube_count_plan.resolve(),
        input,
        output,
        launch_info.cube_count_plan.as_args(),
        launch_info.blueprint,
        dtypes,
    )
}
