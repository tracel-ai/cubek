use super::super::{decompose_linear, shape_divmod};
use crate::definition::{AdaptiveAvgPoolOptions, PoolError};
use crate::kernel::forward::{Position, view4d};
use cubecl::{
    CubeDim, Runtime, calculate_cube_count_elemwise,
    num_traits::Zero,
    prelude::TensorBinding,
    prelude::*,
    std::{FastDivmod, tensor::View},
    tensor_vector_size_parallel,
};

#[cube(launch, address_type = "dynamic")]
fn adaptive_avg_pool2d_backward_direct<E: Numeric, N: Size>(
    grad: &Tensor<Vector<E, N>>,
    output: &mut View<Vector<E, N>, Position, ReadWrite>,
    out_shape: Sequence<FastDivmod<usize>>,
    working_units: usize,
    #[define(E)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= working_units {
        terminate!();
    }

    let (_, out_h, out_w, _) = output.shape();
    let (grad_stride_h, grad_stride_w) = (grad.stride(1), grad.stride(2));
    let (grad_h, grad_w) = (grad.shape(1), grad.shape(2));

    let (b, ih, iw, c) = decompose_linear(ABSOLUTE_POS * output.vector_size(), &out_shape);

    let oh_start = start_index(ih, out_h, grad_h);
    let oh_end = end_index(ih, out_h, grad_h);

    let ow_start = start_index(iw, out_w, grad_w);
    let ow_end = end_index(iw, out_w, grad_w);

    let mut grad_acc = Vector::zero();

    let index_base = b * grad.stride(0) + (c * grad.stride(3));

    for oh in oh_start..oh_end {
        let ih_start = start_index(oh, grad_h, out_h);
        let ih_end = end_index(oh, grad_h, out_h);

        if ih >= ih_start && ih < ih_end {
            for ow in ow_start..ow_end {
                let iw_start = start_index(ow, grad_w, out_w);
                let iw_end = end_index(ow, grad_w, out_w);

                if iw >= iw_start && iw < iw_end {
                    let num_ih = ih_end - ih_start;
                    let num_iw = iw_end - iw_start;

                    let index = index_base + (oh * grad_stride_h) + (ow * grad_stride_w);
                    grad_acc +=
                        grad[index / grad.vector_size()] / Vector::cast_from(num_iw * num_ih);
                }
            }
        }
    }

    output[(b, ih, iw, c)] = grad_acc;
}

#[cube]
fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    (output_size_index * input_size) / output_size
}

#[cube]
fn end_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    let index = (output_size_index + 1) * input_size;
    let index = index.div_ceil(output_size);

    if input_size < index {
        input_size
    } else {
        index
    }
}

pub(crate) fn adaptive_avg_pool2d_backward_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    out_grad: TensorBinding<R>,
    output: TensorBinding<R>,
    _options: AdaptiveAvgPoolOptions<2>,
    dtype: StorageType,
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

    adaptive_avg_pool2d_backward_direct::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        vector_size,
        out_grad.into_tensor_arg(),
        view4d(output.clone(), vector_size),
        shape_divmod(&output),
        working_units,
        dtype,
    );

    Ok(())
}
