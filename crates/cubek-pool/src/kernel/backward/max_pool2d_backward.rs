use super::super::{decompose_linear, shape_divmod};
use crate::{
    definition::{MaxPoolOptions, PoolError},
    kernel::backward::{PoolBackwardArgs, PoolBackwardArgsLaunch},
};
use cubecl::{
    CubeDim, Runtime, calculate_cube_count_elemwise, num_traits::Zero, prelude::TensorBinding,
    prelude::*, std::FastDivmod, tensor_vector_size_parallel,
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn max_pool2d_with_indices_backward_kernel<E: Numeric, I: Int, N: Size>(
    grad: &Tensor<Vector<E, N>>,
    indices: &Tensor<Vector<I, N>>,
    output: &mut Tensor<Vector<E, N>>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[define(E, I)] _dtypes: [StorageType; 2],
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (batch, ih, iw, channel) =
        decompose_linear(ABSOLUTE_POS * output.vector_size(), &out_shape);

    let vector_size = grad.vector_size();

    let index_current = ih * output.shape(2) + iw;

    let (oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        ih as i32,
        iw as i32,
        grad.shape(1) as u32,
        grad.shape(2) as u32,
        args,
        kernel_size_0,
        kernel_size_1,
    );

    let mut grad_acc = Vector::zero();

    let grad_idx_base = batch * grad.stride(0) + channel * grad.stride(3);
    let ind_idx_base = batch * indices.stride(0) + channel * indices.stride(3);

    for oh in oh_start..oh_end {
        for ow in ow_start..ow_end {
            let grad_index =
                grad_idx_base + oh as usize * grad.stride(1) + ow as usize * grad.stride(2);
            let indices_index =
                ind_idx_base + oh as usize * indices.stride(1) + ow as usize * indices.stride(2);
            let index_max = Vector::<u32, N>::cast_from(indices[indices_index / vector_size]);

            grad_acc += select_many(
                index_max.equal(Vector::cast_from(index_current)),
                grad[grad_index / vector_size],
                Vector::zero(),
            );
        }
    }

    let index_output = batch * output.stride(0)
        + ih * output.stride(1)
        + iw * output.stride(2)
        + channel * output.stride(3);

    output[index_output / output.vector_size()] = grad_acc;
}

#[cube]
fn loop_ranges(
    ih: i32,
    iw: i32,
    grad_h: u32,
    grad_w: u32,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
) -> (u32, u32, u32, u32) {
    let kms_0 = args.dilation_0 * kernel_size_0 - args.stride_0;
    let kms_1 = args.dilation_1 * kernel_size_1 - args.stride_1;

    let oh_start = clamp_min((ih + args.padding_0 - kms_0) / args.stride_0, 0) as u32;
    let ow_start = clamp_min((iw + args.padding_1 - kms_1) / args.stride_1, 0) as u32;
    let oh_end = clamp_max(clamp_min(kms_0, 0) as u32 + oh_start, grad_h - 1) + 1;
    let ow_end = clamp_max(clamp_min(kms_1, 0) as u32 + ow_start, grad_w - 1) + 1;

    (oh_start, oh_end, ow_start, ow_end)
}

#[allow(clippy::too_many_arguments)]
pub(crate) fn max_pool2d_with_indices_backward_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    indices: TensorBinding<R>,
    output: TensorBinding<R>,
    options: MaxPoolOptions<2>,
    dtype: StorageType,
    indices_dtype: StorageType,
) -> Result<(), PoolError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        max_pool2d_with_indices_backward_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            out_grad.into_tensor_arg(),
            indices.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            shape_divmod(&output),
            working_units,
            PoolBackwardArgsLaunch::new(
                options.window.stride[0] as i32,
                options.window.stride[1] as i32,
                options.dilation[0] as i32,
                options.dilation[1] as i32,
                options.window.padding[0] as i32,
                options.window.padding[1] as i32,
            ),
            options.window.kernel_size[0] as i32,
            options.window.kernel_size[1] as i32,
            [dtype, indices_dtype],
        )
    };

    Ok(())
}
