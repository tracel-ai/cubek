use cubek_test_utils::HostData;

use super::{build_f32_output, for_each_output_coord, output_shape};

pub fn reference_mean(input: &HostData, axis: usize) -> HostData {
    let axis_len = input.shape[axis];
    let out_shape_vec = output_shape(&input.shape, axis);
    let mut data = vec![0.0f32; out_shape_vec.iter().product()];
    let denom = axis_len as f32;

    for_each_output_coord(&out_shape_vec, |linear, out_coord| {
        let mut coord = out_coord.to_vec();
        let mut acc = 0.0f32;
        for i in 0..axis_len {
            coord[axis] = i;
            acc += input.get_f32(&coord);
        }
        data[linear] = acc / denom;
    });

    build_f32_output(input, axis, data)
}
