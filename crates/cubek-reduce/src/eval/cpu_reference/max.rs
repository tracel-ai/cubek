use cubek_test_utils::{HostData, Progress};

use super::{
    best_of, build_f32_output, bump, for_each_output_coord, output_shape, row_along,
    should_replace_max,
};

pub fn reference_max(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![f32::NEG_INFINITY; out_shape_vec.iter().product()];

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        data[linear] = best_of(
            row_along(input, axis, out_coord),
            f32::NEG_INFINITY,
            should_replace_max,
        )
        .1;
        bump(progress);
    });

    build_f32_output(input, axis, data)
}
