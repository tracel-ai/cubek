//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};
use cubecl::tensor_line_size_parallel;

use crate::components::batch::naive::{NaiveBatchMatmulFamily, NaiveBlueprint};
use crate::definition::{CubeCountPlan, MatmulLineSizes};
use crate::definition::{MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError};

use crate::components::batch::BatchMatmulFamily;
use crate::launch::InputArg;
use crate::launch::handle::{MatmulInputHandle, MatmulInputHandleRef};
use crate::launch::{ConcreteInputsFactory, ConcreteOutputFactory, OutputArg, TensorArgs};
use crate::routines::naive::NaiveRoutine;

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
    let (cube_dim_x, cube_dim_y) = (32, 8);
    let rank = lhs.shape().len();
    let dim1 = rank - 1;
    let dim2 = rank - 2;

    let lhs_layout = matrix_batch_layout(lhs.data().strides);
    let rhs_layout = matrix_batch_layout(rhs.data().strides);

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
    );

    let blueprint = NaiveBlueprint {};
    let config =
        NaiveBatchMatmulFamily::expand_config(client, &problem, &blueprint, &line_sizes, dtypes)?;

    let cube_count_plan =
        simple_cube_count(lhs_shape, rhs_shape, out_shape, cube_dim_x, cube_dim_y)?;

    let input = <InputArg<TensorArgs> as ConcreteInputsFactory<NaiveRoutine>>::create(
        client,
        &lhs,
        &rhs,
        &blueprint,
        &problem,
        &line_sizes,
        config,
        dtypes,
    );
    let output = <OutputArg<TensorArgs> as ConcreteOutputFactory<NaiveRoutine>>::create(
        client,
        out,
        &blueprint,
        &problem,
        &line_sizes,
        config,
        dtypes,
    );

    let result = unsafe {
        NaiveBatchMatmulFamily::launch_unchecked::<TensorArgs, R>(
            client,
            CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
            cube_count_plan.resolve(),
            input,
            output,
            cube_count_plan.as_args(),
            config,
            dtypes,
        )
    };

    match result {
        Ok(_) => Ok(()),
        Err(err) => Err(MatmulSetupError::Launch(err)),
    }
}

#[allow(clippy::result_large_err)]
fn simple_cube_count(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    cube_dim_x: usize,
    cube_dim_y: usize,
) -> Result<CubeCountPlan, MatmulSetupError> {
    let ndims = lhs_shape.len();
    let num_rows = lhs_shape[ndims - 2];
    let num_cols = rhs_shape[ndims - 1];

    let m_cubes = f32::ceil(num_rows as f32 / cube_dim_x as f32) as u32;
    let n_cubes = f32::ceil(num_cols as f32 / cube_dim_y as f32) as u32;
    let mut batch_cubes = 1u32;

    #[allow(clippy::needless_range_loop)]
    for i in 0..ndims - 2 {
        batch_cubes *= output_shape[i] as u32;
    }

    let cube_count_plan = CubeCountPlan::FromProblem {
        m_cubes,
        n_cubes,
        batch_cubes,
    };
    let max_cube_count = u16::MAX as u32;

    if m_cubes > max_cube_count || n_cubes > max_cube_count || batch_cubes > max_cube_count {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CubeCountTooBig(cube_count_plan.resolve()),
        ));
    }

    Ok(cube_count_plan)
}
