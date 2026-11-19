use cubek_test_utils::{HostData, Progress};

use super::{build_f32_output, bump, for_each_output_coord, output_shape, row_along};

/// Logical-AND reference: 1.0 if every element along `axis` is non-zero, else 0.0.
pub fn reference_all(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![1.0f32; out_shape_vec.iter().product()];

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        data[linear] = f32::from(u8::from(
            row_along(input, axis, out_coord).all(|value| value != 0.0),
        ));
        bump(progress);
    });

    build_f32_output(input, axis, data)
}
