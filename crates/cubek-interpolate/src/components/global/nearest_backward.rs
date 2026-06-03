use crate::definition::NearestMode;
use cubecl::prelude::*;
use cubecl::{
    ir::StorageType,
    num_traits::Zero,
    std::FastDivmod,
    std::tensor::layout::{linear::LinearLayout, *},
};

#[cube(launch_unchecked, address_type = "dynamic")]
pub fn execute_interpolate_nearest_backward<F: Float, N: Size>(
    grad: &Tensor<Vector<F, N>>,
    output: &mut Tensor<Vector<F, N>>,
    shape_out: Sequence<FastDivmod<usize>>,
    out_layout: LinearLayout,
    #[comptime] nearest_mode: NearestMode,
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

    let grad_y_start = start_index::<F>(out_y, grad_h, out_h, nearest_mode);
    let grad_y_end = end_index::<F>(out_y, grad_h, out_h, nearest_mode);
    let grad_x_start = start_index::<F>(out_x, grad_w, out_w, nearest_mode);
    let grad_x_end = end_index::<F>(out_x, grad_w, out_w, nearest_mode);

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
fn start_index<F: Float>(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    #[comptime] nearest_mode: NearestMode,
) -> usize {
    match nearest_mode {
        NearestMode::Floor => {
            let numerator = F::cast_from(input_index * output_size);
            let div = (numerator / F::cast_from(input_size)).ceil();
            usize::cast_from(div)
        }
        NearestMode::Exact => {
            let num = F::cast_from(input_index * output_size);
            let den = F::cast_from(input_size);
            let div = (num / den).ceil() - F::new(0.5);

            let mask = F::cast_from((div >= F::zero()) as usize);
            usize::cast_from(div.ceil() * mask)
        }
    }
}

#[cube]
fn end_index<F: Float>(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    #[comptime] nearest_mode: NearestMode,
) -> usize {
    start_index::<F>(input_index + 1, output_size, input_size, nearest_mode)
}
