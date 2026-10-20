use crate::definition::InterpolateError;
use cubecl::{
    calculate_cube_count_elemwise,
    client::Client,
    num_traits::Zero,
    prelude::*,
    std::FastDivmod,
    std::tensor::layout::{linear::LinearLayout, *},
    tensor_vector_size_parallel,
};

use super::backward::{linear_layout, shape_divmod};

pub(crate) fn interpolate_bilinear_backward_launch(
    client: &Client,
    out_grad: TensorBinding,
    input_grad: TensorBinding,
    align_corners: bool,
    dtype: ElemType,
) -> Result<(), InterpolateError> {
    let vector_size = [&out_grad, &input_grad]
        .into_iter()
        .map(|binding| {
            tensor_vector_size_parallel(
                client.io_optimized_vector_sizes(dtype.size()),
                &binding.shape,
                &binding.strides,
                3,
            )
        })
        .min()
        .unwrap();
    let shape = shape_divmod(&input_grad);
    let layout = linear_layout(&input_grad, vector_size);
    let working_units = input_grad.shape.iter().product::<usize>() / vector_size;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);
    let address_type = out_grad
        .required_address_type(dtype.size())
        .max(input_grad.required_address_type(dtype.size()));

    unsafe {
        execute_interpolate_bilinear_backward::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            out_grad.into_tensor_arg(),
            input_grad.into_tensor_arg(),
            shape,
            layout,
            align_corners,
            dtype,
        )
    };
    Ok(())
}

/// Each input pixel gathers its contributing output gradients. No atomics or
/// pre-zeroed output buffer are needed, and the loops stay small for upsampling.
#[cube(launch_unchecked, address_type = "dynamic")]
fn execute_interpolate_bilinear_backward<F: Float, N: Size>(
    grad: &Tensor<Vector<F, N>>,
    input_grad: &mut Tensor<Vector<F, N>>,
    input_shape: Sequence<FastDivmod<usize>>,
    input_layout: LinearLayout,
    #[comptime] align_corners: bool,
    #[define(F)] _dtype: ElemType,
) {
    if ABSOLUTE_POS >= input_grad.len() {
        terminate!();
    }

    let vector_size = grad.vector_size();
    let input_idx = input_layout.to_source_pos(ABSOLUTE_POS);
    let (rem, channel) = input_shape[3].div_mod(ABSOLUTE_POS * vector_size);
    let (rem, x) = input_shape[2].div_mod(rem);
    let (batch, y) = input_shape[1].div_mod(rem);
    let height = input_grad.shape(1);
    let width = input_grad.shape(2);
    let grad_height = grad.shape(1);
    let grad_width = grad.shape(2);
    let (y_start, y_end) = contributing_range(y, height, grad_height, align_corners);
    let (x_start, x_end) = contributing_range(x, width, grad_width, align_corners);
    let base = batch * grad.stride(0) + channel * grad.stride(3);

    let mut sum = Vector::<F, N>::zero();
    for grad_y in y_start..y_end {
        let wy = sample_weight::<F>(y, grad_y, height, grad_height, align_corners);
        for grad_x in x_start..x_end {
            let wx = sample_weight::<F>(x, grad_x, width, grad_width, align_corners);
            let index = base + grad_y * grad.stride(1) + grad_x * grad.stride(2);
            sum += grad[index / vector_size] * Vector::cast_from(wy * wx);
        }
    }
    input_grad[input_idx] = sum;
}

/// Invert the support interval (input - 1, input + 1) using integer arithmetic.
/// The start may include one zero-weight sample. Clamped edge samples belong to
/// the first/last pixel, and singleton axes must include the entire gradient.
#[cube]
fn contributing_range(
    input: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] align_corners: bool,
) -> (usize, usize) {
    let mut start = 0;
    let mut end = output_size;
    if input_size > 1 && output_size > 1 {
        if align_corners {
            let denominator = input_size - 1;
            if input > 0 {
                start = (input - 1) * (output_size - 1) / denominator;
            }
            if input + 1 < input_size {
                let numerator = (input + 1) * (output_size - 1);
                end = (numerator + denominator - 1) / denominator;
            }
        } else {
            let denominator = 2 * input_size;
            if input > 0 {
                let scaled = (2 * input - 1) * output_size;
                if scaled > input_size {
                    start = (scaled - input_size) / denominator;
                }
            }
            if input + 1 < input_size {
                let scaled = (2 * input + 3) * output_size;
                end = 0;
                if scaled > input_size {
                    end = (scaled - input_size + denominator - 1) / denominator;
                }
            }
        }
    }
    (clamp_max(start, output_size), clamp_max(end, output_size))
}

/// Use the forward transform's rational phase rather than subtracting a large
/// floating-point coordinate from the input index. Clamp both taps separately
/// so duplicated taps on singleton axes and at the border retain unit weight.
#[cube]
fn sample_weight<F: Float>(
    input: usize,
    output: usize,
    input_size: usize,
    output_size: usize,
    #[comptime] align_corners: bool,
) -> F {
    let mut numerator = output * (input_size - 1);
    let mut denominator = clamp_min(output_size - 1, 1);
    if !align_corners {
        let scaled = (2 * output + 1) * input_size;
        if scaled < output_size {
            numerator = 0;
        } else {
            numerator = scaled - output_size;
        }
        denominator = 2 * output_size;
    }

    let lower = clamp_max(numerator / denominator, input_size - 1);
    let upper = clamp_max(lower + 1, input_size - 1);
    let fraction = F::cast_from(numerator % denominator) / F::cast_from(denominator);
    let mut weight = F::zero();
    if input == lower {
        weight += F::new(1.0) - fraction;
    }
    if input == upper {
        weight += fraction;
    }
    weight
}
