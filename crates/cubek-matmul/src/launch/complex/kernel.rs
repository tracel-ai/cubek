use cubecl::prelude::*;

/// A `C32` element as the kernels read it: its real and imaginary parts, side by side.
type C32Parts = Vector<f32, Const<2>>;

#[cube(launch_unchecked)]
pub(super) fn extract_c32_part_kernel(
    input_meta: &Tensor<f32>,
    input_parts: &[C32Parts],
    out_parts: &mut Tensor<f32>,
    #[comptime] part: u32,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS >= out_parts.len() {
        terminate!();
    }
    let mut input_offset = 0usize;
    #[unroll]
    for axis in 0..rank {
        let coord = out_parts.coordinate(ABSOLUTE_POS, axis);
        input_offset += coord * input_meta.stride(axis);
    }
    let value = input_parts[input_offset];
    out_parts[ABSOLUTE_POS] = if part == 0 {
        value.extract(0usize)
    } else {
        value.extract(1usize)
    };
}

#[cube(launch_unchecked)]
pub(super) fn compose_c32_parts_kernel(
    out_meta: &Tensor<f32>,
    out_parts: &mut [C32Parts],
    real_pos: &Tensor<f32>,
    real_neg: &Tensor<f32>,
    imag_left: &Tensor<f32>,
    imag_right: &Tensor<f32>,
    #[comptime] lhs_imag_sign: i32,
    #[comptime] rhs_imag_sign: i32,
    #[comptime] rank: usize,
) {
    if ABSOLUTE_POS >= real_pos.len() {
        terminate!();
    }
    let mut out_offset = 0usize;
    let mut dense_offset = 0usize;
    #[unroll]
    for axis in 0..rank {
        let coord = real_pos.coordinate(ABSOLUTE_POS, axis);
        out_offset += coord * out_meta.stride(axis);
        dense_offset += coord * real_pos.stride(axis);
    }
    let lhs_sign = f32::cast_from(lhs_imag_sign);
    let rhs_sign = f32::cast_from(rhs_imag_sign);
    let real = real_pos[dense_offset] - lhs_sign * rhs_sign * real_neg[dense_offset];
    let imag = rhs_sign * imag_left[dense_offset] + lhs_sign * imag_right[dense_offset];
    let mut value = Vector::<f32, Const<2>>::empty();
    value.insert(0usize, real);
    value.insert(1usize, imag);
    out_parts[out_offset] = value;
}
