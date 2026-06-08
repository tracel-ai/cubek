use cubek_test_utils::{HostData, Progress};

use super::{build_f32_output, for_each_output_coord, output_shape};

/// Logical-AND reference: 1.0 if every element along `axis` is non-zero, else 0.0.
pub fn reference_all(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    let axis_len = input.shape[axis];
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![1.0f32; out_shape_vec.iter().product()];

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        let mut coord = out_coord.to_vec();
        let mut all = true;
        for i in 0..axis_len {
            coord[axis] = i;
            if input.get_f32(&coord) == 0.0 {
                all = false;
                break;
            }
        }
        data[linear] = if all { 1.0 } else { 0.0 };
        if let Some(p) = progress {
            p.bump();
        }
    });

    build_f32_output(input, axis, data)
}
