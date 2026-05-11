use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};
use std::f32::consts::PI;

use super::super::{contiguous_strides, for_each_output_coord};

fn sinc(x: f32) -> f32 {
    if x.abs() < 1e-8 {
        1.0
    } else {
        let pi_x = x * PI;
        pi_x.sin() / pi_x
    }
}

fn lanczos3_weight(x: f32) -> f32 {
    let x = x.abs();
    if x < 3.0 {
        sinc(x) * sinc(x / 3.0)
    } else {
        0.0
    }
}

pub fn reference_lanczos3(
    input: &HostData,
    output_shape: &[usize],
    align_corners: bool,
    progress: Option<&Progress>,
) -> HostData {
    let mut data = vec![0.0f32; output_shape.iter().product()];
    let input_h = input.shape[1] as i32;
    let input_w = input.shape[2] as i32;
    let input_height_f = (input.shape[1] - 1) as f32;
    let input_width_f = (input.shape[2] - 1) as f32;

    for_each_output_coord(output_shape, |linear, out_coord| {
        let b = out_coord[0];
        let y_out = out_coord[1];
        let x_out = out_coord[2];
        let c = out_coord[3];

        let fy = if align_corners {
            let denominator = (output_shape[1] - 1).max(1) as f32;
            y_out as f32 * (input_height_f / denominator)
        } else {
            let in_size = input.shape[1] as f32;
            let out_size = output_shape[1] as f32;
            (y_out as f32 + 0.5) * (in_size / out_size) - 0.5
        };

        let fx = if align_corners {
            let denominator = (output_shape[2] - 1).max(1) as f32;
            x_out as f32 * (input_width_f / denominator)
        } else {
            let in_size = input.shape[2] as f32;
            let out_size = output_shape[2] as f32;
            (x_out as f32 + 0.5) * (in_size / out_size) - 0.5
        };

        let y_int = fy.floor() as i32;
        let x_int = fx.floor() as i32;

        let mut val = 0.0;
        let mut weight_sum = 0.0;

        for j in 0..6 {
            let row_idx = y_int - 2 + j;

            if row_idx >= 0 && row_idx < input_h {
                let wy = lanczos3_weight(fy - (y_int - 2 + j) as f32);

                for i in 0..6 {
                    let col_idx = x_int - 2 + i;

                    if col_idx >= 0 && col_idx < input_w {
                        let wx = lanczos3_weight(fx - (x_int - 2 + i) as f32);
                        let weight = wy * wx;

                        val += input.get_f32(&[b, row_idx as usize, col_idx as usize, c]) * weight;
                        weight_sum += weight;
                    }
                }
            }
        }

        data[linear] = if weight_sum != 0.0 {
            val / weight_sum
        } else {
            0.0
        };

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
