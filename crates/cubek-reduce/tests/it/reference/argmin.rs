use cubek_test_utils::HostData;

use super::{build_f32_output, for_each_output_coord, output_shape};

pub fn reference_argmin(input: &HostData, axis: usize) -> HostData {
    let axis_len = input.shape[axis];
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![0.0f32; out_shape_vec.iter().product()];

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        let mut coord = out_coord.to_vec();
        let mut best = f32::INFINITY;
        let mut best_idx: u32 = 0;
        for i in 0..axis_len {
            coord[axis] = i;
            let v = input.get_f32(&coord);
            if v < best {
                best = v;
                best_idx = i as u32;
            }
        }
        data[linear] = best_idx as f32;
    });

    build_f32_output(input, axis, data)
}
