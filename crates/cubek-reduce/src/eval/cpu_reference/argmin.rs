use cubek_test_utils::{HostData, Progress};

use super::{
    best_of, build_f32_output, bump, for_each_output_coord, output_shape, row_along,
    should_replace_min,
};

pub fn reference_argmin(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![0.0f32; out_shape_vec.iter().product()];

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        data[linear] = best_of(
            row_along(input, axis, out_coord),
            f32::INFINITY,
            should_replace_min,
        )
        .0 as f32;
        bump(progress);
    });

    build_f32_output(input, axis, data)
}
