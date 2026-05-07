use crate::cpu_reference::forward::get_pixel_fraction;
use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use super::super::{contiguous_strides, for_each_output_coord};

pub fn reference_bilinear(
    input: &HostData,
    output_shape: &[usize],
    align_corners: bool,
    progress: Option<&Progress>,
) -> HostData {
    let mut data = vec![0.0f32; output_shape.iter().product()];

    for_each_output_coord(output_shape, |linear, out_coord| {
        let b = out_coord[0];
        let y_out = out_coord[1];
        let x_out = out_coord[2];
        let c = out_coord[3];

        let fy = get_pixel_fraction(input.shape[1], output_shape[1], y_out, align_corners)
            .clamp(0.0, (input.shape[1] - 1) as f32);
        let fx = get_pixel_fraction(input.shape[2], output_shape[2], x_out, align_corners)
            .clamp(0.0, (input.shape[2] - 1) as f32);

        let y0 = fy.floor() as usize;
        let x0 = fx.floor() as usize;
        let y1 = (y0 + 1).min(input.shape[1] - 1);
        let x1 = (x0 + 1).min(input.shape[2] - 1);

        let dy = fy - y0 as f32;
        let dx = fx - x0 as f32;

        let v00 = input.get_f32(&[b, y0, x0, c]);
        let v01 = input.get_f32(&[b, y0, x1, c]);
        let v10 = input.get_f32(&[b, y1, x0, c]);
        let v11 = input.get_f32(&[b, y1, x1, c]);

        let v0 = v00 * (1.0 - dx) + v01 * dx;
        let v1 = v10 * (1.0 - dx) + v11 * dx;
        let value = v0 * (1.0 - dy) + v1 * dy;

        data[linear] = value;

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
