use cubek_test_utils::{HostData, Progress};

use super::{build_f32_output, for_each_output_coord, output_shape};

/// Logical-OR reference: 1.0 if any element along `axis` is non-zero, else 0.0.
pub fn reference_any(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    let axis_len = input.shape[axis];
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![0.0f32; out_shape_vec.iter().product()];

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        let mut coord = out_coord.to_vec();
        let mut any = false;
        for i in 0..axis_len {
            coord[axis] = i;
            if input.get_f32(&coord) != 0.0 {
                any = true;
                break;
            }
        }
        data[linear] = if any { 1.0 } else { 0.0 };
        if let Some(p) = progress {
            p.bump();
        }
    });

    build_f32_output(input, axis, data)
}
