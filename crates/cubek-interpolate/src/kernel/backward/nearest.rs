use super::super::{linear_layout, shape_divmod};
use crate::InterpolateError;
use cubecl::{calculate_cube_count_elemwise, prelude::*};
use cubecl::{
    num_traits::Zero,
    std::{
        FastDivmod,
        tensor::layout::{linear::LinearLayout, *},
    },
    tensor_vector_size_parallel,
};

#[cube(launch_unchecked, address_type = "dynamic")]
fn interpolate_nearest_backward_kernel<F: Float, N: Size>(
    grad: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[define(F)] _dtype: StorageType,
) {
    if ABSOLUTE_POS >= output.len() {
        terminate!();
    }

    let vector_size = grad.vector_size();
    let out_idx = out_layout.to_source_pos(ABSOLUTE_POS);

    let out_h = output.shape(1);
    let out_w = output.shape(2);
    let grad_h = grad.shape(1);
    let grad_w = grad.shape(2);

    let (rem, c) = shape_out[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, out_x) = shape_out[2].div_mod(rem);
    let (b, out_y) = shape_out[1].div_mod(rem);

    let grad_y_start = start_index::<F>(out_y, grad_h, out_h);
    let grad_y_end = end_index::<F>(out_y, grad_h, out_h);
    let grad_x_start = start_index::<F>(out_x, grad_w, out_w);
    let grad_x_end = end_index::<F>(out_x, grad_w, out_w);

    let index_grad_base = b * grad.stride(0) + c * grad.stride(3);

    let mut sum = Vector::zero();

    for grad_y in grad_y_start..grad_y_end {
        for grad_x in grad_x_start..grad_x_end {
            let index_grad = index_grad_base + grad_y * grad.stride(1) + grad_x * grad.stride(2);

            sum += grad[index_grad / vector_size];
        }
    }

    output[out_idx] = sum;
}

#[cube]
fn start_index<F: Float>(input_index: usize, output_size: usize, input_size: usize) -> usize {
    let numerator = F::cast_from(input_index * output_size);
    let div = (numerator / F::cast_from(input_size)).ceil();

    usize::cast_from(div)
}

#[cube]
fn end_index<F: Float>(input_index: usize, output_size: usize, input_size: usize) -> usize {
    let numerator = F::cast_from((input_index + 1) * output_size);
    let div = (numerator / F::cast_from(input_size)).ceil();
    let index = usize::cast_from(div);

    clamp_max(index, output_size)
}

pub(crate) fn interpolate_nearest_backward_launch<R: Runtime>(
    client: &ComputeClient<R>,
    out_grad: TensorBinding<R>,
    output: TensorBinding<R>,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &out_grad.shape,
        &out_grad.strides,
        out_grad.shape.len() - 1,
    );
    let shape_out = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = out_grad
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        interpolate_nearest_backward_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            out_grad.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            shape_out,
            out_layout,
            dtype,
        )
    };

    Ok(())
}
