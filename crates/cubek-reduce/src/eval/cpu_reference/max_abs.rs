use cubek_test_utils::{HostData, Progress};

use super::{build_f32_output, for_each_output_coord, output_shape};

pub fn reference_max_abs(input: &HostData, axis: usize, progress: Option<&Progress>) -> HostData {
    let axis_len = input.shape[axis];
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![0.0f32; out_shape_vec.iter().product()];

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        let mut coord = out_coord.to_vec();
        let mut best = 0.0f32;
        for i in 0..axis_len {
            coord[axis] = i;
            let v = input.get_f32(&coord).abs();
            if v > best {
                best = v;
            }
        }
        data[linear] = best;
        if let Some(p) = progress {
            p.bump();
        }
    });

    build_f32_output(input, axis, data)
}
