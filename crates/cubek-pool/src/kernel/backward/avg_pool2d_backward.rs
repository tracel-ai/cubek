use core::result::Result::Ok;

use super::super::{decompose_linear, shape_divmod};
use crate::definition::{AvgPoolOptions, PoolError};
use crate::kernel::forward::{Position, view4d};
use cubecl::{
    CubeDim, Runtime, calculate_cube_count_elemwise,
    num_traits::Zero,
    prelude::TensorBinding,
    prelude::*,
    std::{FastDivmod, tensor::View},
    tensor_vector_size_parallel,
};

#[derive(CubeLaunch, CubeType)]
pub(crate) struct PoolBackwardArgs {
    pub stride_0: i32,
    pub stride_1: i32,
    pub dilation_0: i32,
    pub dilation_1: i32,
    pub padding_0: i32,
    pub padding_1: i32,
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn avg_pool2d_backward_kernel<E: Numeric, N: Size>(
    grad: &Tensor<Vector<E, N>>,
    output: &mut View<Vector<E, N>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    args: &PoolBackwardArgs,
    #[comptime] kernel_size_0: i32,
    #[comptime] kernel_size_1: i32,
    #[comptime] count_include_pad: bool,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let vector_size = grad.vector_size();

    let (batch, ih, iw, channel) =
        decompose_linear(ABSOLUTE_POS * output.vector_size(), &out_shape);

    let mut grad_acc = Vector::zero();

    let (oh_start, oh_end, ow_start, ow_end) = loop_ranges(
        ih as i32,
        iw as i32,
        grad.shape(1) as u32,
        grad.shape(2) as u32,
        args,
        kernel_size_0,
        kernel_size_1,
    );

    let padding_0 = args.padding_0 as u32;
    let padding_1 = args.padding_1 as u32;
    let stride_0 = args.stride_0 as u32;
    let stride_1 = args.stride_1 as u32;
    let kernel_size_0 = comptime![kernel_size_0 as u32];
    let kernel_size_1 = comptime![kernel_size_1 as u32];

    let (_, out_h, out_w, _) = output.shape();
    let index_base = batch * grad.stride(0) + channel * grad.stride(3);
    let border_bottom = out_h as u32 + padding_0;
    let border_right = out_w as u32 + padding_1;
    let begin_h = ih as u32 + padding_0;
    let begin_w = iw as u32 + padding_1;

    for oh in oh_start..oh_end {
        let ih_start = oh * stride_0;
        let ih_end = clamp_max(ih_start + kernel_size_0, border_bottom);
        let ih_start = clamp_min(ih_start, padding_0);

        if begin_h >= ih_start && (ih as u32) < ih_end {
            for ow in ow_start..ow_end {
                let index =
                    index_base + oh as usize * grad.stride(1) + ow as usize * grad.stride(2);

                let iw_start = ow * stride_1;
                let iw_end = clamp_max(iw_start + kernel_size_1, border_right);
                let iw_start = clamp_min(iw_start, padding_1);

                if begin_w >= iw_start && (iw as u32) < iw_end {
                    if count_include_pad {
                        grad_acc += grad[index / vector_size]
                            / Vector::cast_from(kernel_size_0 * kernel_size_1);
                    } else {
                        let ih_diff = ih_end - ih_start;
                        let iw_diff = iw_end - iw_start;
                        let count = Vector::cast_from(ih_diff * iw_diff);
                        grad_acc += grad[index / vector_size] / count;
                    }
                }
            }
        }
    }

    output[(batch, ih, iw, channel)] = grad_acc;
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

pub(crate) fn avg_pool2d_backward_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    output: TensorBinding<R>,
    options: AvgPoolOptions<2>,
    dtype: StorageType,
) -> Result<(), PoolError> {
    let dilation = 1;

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
        avg_pool2d_backward_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            out_grad.into_tensor_arg(),
            view4d(output.clone(), vector_size),
            shape_divmod(&output),
            working_units,
            PoolBackwardArgsLaunch::new(
                options.window.stride[0] as i32,
                options.window.stride[1] as i32,
                dilation,
                dilation,
                options.window.padding[0] as i32,
                options.window.padding[1] as i32,
            ),
            options.window.kernel_size[0] as i32,
            options.window.kernel_size[1] as i32,
            options.count_include_pad,
            dtype,
        )
    };

    Ok(())
}
