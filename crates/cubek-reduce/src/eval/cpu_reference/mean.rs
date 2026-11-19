use cubek_test_utils::{HostData, Progress};

use super::{build_f32_output, bump, for_each_output_coord, output_shape, row_along};

pub fn reference_mean(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    let axis_len = input.shape[axis];
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![0.0f32; out_shape_vec.iter().product()];
    let denom = axis_len as f32;

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        data[linear] =
            row_along(input, axis, out_coord).fold(0.0, |acc, value| acc + value) / denom;
        bump(progress);
    });

    build_f32_output(input, axis, data)
}
