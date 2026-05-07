use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use super::super::{contiguous_strides, for_each_output_coord};

fn cubic_convolution_1(x: f32, a: f32) -> f32 {
    ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0
}

fn cubic_convolution_2(x: f32, a: f32) -> f32 {
    ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a
}

fn cubic_interp_1d(x0: f32, x1: f32, x2: f32, x3: f32, t: f32) -> f32 {
    let a = -0.75;
    let coeffs0 = cubic_convolution_2(t + 1.0, a);
    let coeffs1 = cubic_convolution_1(t, a);
    let coeffs2 = cubic_convolution_1(1.0 - t, a);
    let coeffs3 = cubic_convolution_2(2.0 - t, a);

    x0 * coeffs0 + x1 * coeffs1 + x2 * coeffs2 + x3 * coeffs3
}

pub fn reference_bicubic(
    input: &HostData,
    output_shape: &[usize],
    align_corners: bool,
    progress: Option<&Progress>,
) -> HostData {
    let mut data = vec![0.0f32; output_shape.iter().product()];
    let input_height_f = (input.shape[1] - 1) as f32;
    let input_width_f = (input.shape[2] - 1) as f32;

    for_each_output_coord(output_shape, |linear, out_coord| {
        let b = out_coord[0];
        let y_out = out_coord[1];
        let x_out = out_coord[2];
        let c = out_coord[3];

        let frac_y = if align_corners {
            let output_height = (output_shape[1] - 1).max(1) as f32;
            (y_out as f32 * input_height_f) / output_height
        } else {
            let in_size = input.shape[1] as f32;
            let out_size = output_shape[1] as f32;
            (y_out as f32 + 0.5) * (in_size / out_size) - 0.5
        };

        let y_in_f = frac_y.floor();
        let yw = frac_y - y_in_f;

        let y0 = (y_in_f - 1.0).clamp(0.0, input_height_f) as usize;
        let y1 = (y_in_f).clamp(0.0, input_height_f) as usize;
        let y2 = (y_in_f + 1.0).clamp(0.0, input_height_f) as usize;
        let y3 = (y_in_f + 2.0).clamp(0.0, input_height_f) as usize;

        let frac_x = if align_corners {
            let output_width = (output_shape[2] - 1).max(1) as f32;
            (x_out as f32 * input_width_f) / output_width
        } else {
            let in_size = input.shape[2] as f32;
            let out_size = output_shape[2] as f32;
            (x_out as f32 + 0.5) * (in_size / out_size) - 0.5
        };

        let x_in_f = frac_x.floor();
        let xw = frac_x - x_in_f;

        let x0 = (x_in_f - 1.0).clamp(0.0, input_width_f) as usize;
        let x1 = (x_in_f).clamp(0.0, input_width_f) as usize;
        let x2 = (x_in_f + 1.0).clamp(0.0, input_width_f) as usize;
        let x3 = (x_in_f + 2.0).clamp(0.0, input_width_f) as usize;

        let mut row_coeffs = [0.0f32; 4];
        let y_coords = [y0, y1, y2, y3];
        let x_coords = [x0, x1, x2, x3];

        for i in 0..4 {
            let v0 = input.get_f32(&[b, y_coords[i], x_coords[0], c]);
            let v1 = input.get_f32(&[b, y_coords[i], x_coords[1], c]);
            let v2 = input.get_f32(&[b, y_coords[i], x_coords[2], c]);
            let v3 = input.get_f32(&[b, y_coords[i], x_coords[3], c]);
            row_coeffs[i] = cubic_interp_1d(v0, v1, v2, v3, xw);
        }

        let value = cubic_interp_1d(
            row_coeffs[0],
            row_coeffs[1],
            row_coeffs[2],
            row_coeffs[3],
            yw,
        );

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
