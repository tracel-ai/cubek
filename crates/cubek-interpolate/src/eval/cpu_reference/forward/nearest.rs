use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use super::super::{contiguous_strides, for_each_output_coord};

pub fn reference_nearest(
    input: &HostData,
    output_shape: &[usize],
    progress: Option<&Progress>,
) -> HostData {
    let (h_in, w_in) = (input.shape[1], input.shape[2]);
    let (h_out, w_out) = (output_shape[1], output_shape[2]);

    let ratio_h = (h_in as f32) / (h_out as f32);
    let ratio_w = (w_in as f32) / (w_out as f32);

    let mut data = vec![0.0f32; output_shape.iter().product()];

    for_each_output_coord(output_shape, |linear, out_coord| {
        let b = out_coord[0];
        let y_out = out_coord[1];
        let x_out = out_coord[2];
        let c = out_coord[3];

        let y_in_f = ((y_out as f32 + 0.5) * ratio_h - 0.5)
            .floor()
            .clamp(0.0, (h_in - 1) as f32);

        let x_in_f = ((x_out as f32 + 0.5) * ratio_w - 0.5)
            .floor()
            .clamp(0.0, (w_in - 1) as f32);

        let y_in = y_in_f as usize;
        let x_in = x_in_f as usize;

        data[linear] = input.get_f32(&[b, y_in, x_in, c]);

        if let Some(p) = progress {
            p.bump();
        }
    });

    HostData {
        data: HostDataVec::F32(data),
        shape: Shape::from(output_shape.to_vec()),
        strides: contiguous_strides(output_shape),
    }
}
