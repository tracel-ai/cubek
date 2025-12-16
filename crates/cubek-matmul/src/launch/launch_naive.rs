//! Naive matmul kernel implementation
//!
//! Each local unit will compute a single element of the output matrix.
use cubecl::prelude::*;
use cubecl::tensor_line_size_parallel;

use cubecl::std::tensor::{MatrixBatchLayout, matrix_batch_layout};

use crate::definition::MatmulLineSizes;
use crate::definition::{
    MatmulAvailabilityError, MatmulElems, MatmulProblem, MatmulSetupError, MatrixLayout,
};

use crate::launch::MatmulInputHandle;
use crate::launch::MatmulInputHandleRef;

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

    let cube_count = simple_cube_count(lhs_shape, rhs_shape, out_shape, cube_dim_x, cube_dim_y)?;

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

    let problem = MatmulProblem {
        m: out_shape[rank - 2],
        n: out_shape[rank - 1],
        k: lhs_shape[rank - 1],
        lhs_batches: lhs_shape[..rank - 2].to_vec(),
        rhs_batches: rhs_shape[..rank - 2].to_vec(),
        out_batches: out_shape[..rank - 2].to_vec(),
        lhs_strides: lhs.data().strides.to_vec(),
        rhs_strides: rhs.data().strides.to_vec(),
        lhs_layout: MatrixLayout::RowMajor,
        rhs_layout: MatrixLayout::ColMajor,
    };

    // fn view<'a, R: Runtime>(
    //     client: &ComputeClient<R>,
    //     handle: &'a MatmulInputHandleRef<'a, R>,
    //     layout: MatrixLayout,
    //     line_size: u8,
    //     problem: &MatmulProblem,
    // ) -> ViewArg<'a, Coords3d, R> {
    //     // Checks off, other properties are unused
    //     let config = GlobalLayoutConfig {
    //         matrix_layout: layout,
    //         ..Default::default()
    //     };
    //     match handle {
    //         MatmulInputHandleRef::Normal(handle, _dtype) => {
    //             let layout = GlobalLayoutLaunch::from_handle_batched(
    //                 client, handle, problem, line_size, config,
    //             );
    //             ViewArg::new::<GlobalLayout>(handle.as_array_arg(line_size), layout)
    //         }
    //         MatmulInputHandleRef::Quantized {
    //             data,
    //             scale,
    //             shape,
    //             scheme,
    //             ..
    //         } => {
    //             let (data_layout, scales_layout) = GlobalLayoutLaunch::from_quantized_handle(
    //                 client, data, scale, shape, problem, **scheme, line_size, config,
    //             );
    //             let data_view =
    //                 ViewArg::new::<GlobalLayout>(data.as_array_arg(line_size), data_layout);
    //             let scales_view =
    //                 ViewArg::new::<GlobalScaleLayout>(scale.as_array_arg(1), scales_layout);
    //             ViewArg::new_quantized(data_view, scales_view, **scheme)
    //         }
    //     }
    // }

    // let lhs_view = view(
    //     client,
    //     &lhs,
    //     MatrixLayout::RowMajor,
    //     lhs_line_size,
    //     &problem,
    // );
    // let rhs_view = view(
    //     client,
    //     &rhs,
    //     MatrixLayout::ColMajor,
    //     rhs_line_size,
    //     &problem,
    // );

    // let config = NaiveBatchMatmulFamily::setup(client, &problem, &(), &line_sizes, dtypes)?;

    // let inputs = ConcreteInputsFactory::create(
    //     client,
    //     &lhs,
    //     &rhs,
    //     &(),
    //     &problem,
    //     &line_sizes,
    //     config,
    //     dtypes,
    // );

    // let result = unsafe {
    //     batch::naive::naive_matmul::launch_unchecked(
    //         client,
    //         cube_count,
    //         CubeDim::new(cube_dim_x as u32, cube_dim_y as u32, 1),
    //         lhs_view,
    //         rhs_view,
    //         out.as_tensor_arg(1),
    //         *dtypes.lhs_global,
    //         *dtypes.acc_register,
    //         *dtypes.acc_global,
    //     )
    // };

    // match result {
    //     Ok(_) => Ok(()),
    //     Err(err) => Err(MatmulSetupError::Launch(err)),
    // }
    todo!()
}

#[allow(clippy::result_large_err)]
fn simple_cube_count(
    lhs_shape: &[usize],
    rhs_shape: &[usize],
    output_shape: &[usize],
    cube_dim_x: usize,
    cube_dim_y: usize,
) -> Result<CubeCount, MatmulSetupError> {
    let ndims = lhs_shape.len();
    let num_rows = lhs_shape[ndims - 2];
    let num_cols = rhs_shape[ndims - 1];

    let cubes_x = f32::ceil(num_rows as f32 / cube_dim_x as f32) as u32;
    let cubes_y = f32::ceil(num_cols as f32 / cube_dim_y as f32) as u32;
    let mut num_iter = 1u32;

    #[allow(clippy::needless_range_loop)]
    for i in 0..ndims - 2 {
        num_iter *= output_shape[i] as u32;
    }

    let result = CubeCount::Static(cubes_x, cubes_y, num_iter);
    let max_cube_count = u16::MAX as u32;

    if cubes_x > max_cube_count || cubes_y > max_cube_count || num_iter > max_cube_count {
        return Err(MatmulSetupError::Unavailable(
            MatmulAvailabilityError::CubeCountTooBig(result),
        ));
    }

    Ok(result)
}
