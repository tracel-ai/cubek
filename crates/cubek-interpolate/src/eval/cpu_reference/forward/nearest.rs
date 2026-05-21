use crate::definition::NearestMode;
use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use super::super::{contiguous_strides, for_each_output_coord};

pub fn reference_nearest(
    input: &HostData,
    output_shape: &[usize],
    nearest_mode: NearestMode,
    progress: Option<&Progress>,
) -> HostData {
    let (h_in, w_in) = (input.shape[1], input.shape[2]);
    let (h_out, w_out) = (output_shape[1], output_shape[2]);
    let mut data = vec![0.0f32; output_shape.iter().product()];

    for_each_output_coord(output_shape, |linear, out_coord| {
        let b = out_coord[0];

        let y;
        let x;

        match nearest_mode {
            NearestMode::Exact => {
                y = std::cmp::min(((out_coord[1] * 2 + 1) * h_in) / (h_out * 2), h_in - 1);
                x = std::cmp::min(((out_coord[2] * 2 + 1) * w_in) / (w_out * 2), w_in - 1);
            }
            NearestMode::Floor => {
                y = std::cmp::min((out_coord[1] * h_in) / h_out, h_in - 1);
                x = std::cmp::min((out_coord[2] * w_in) / w_out, w_in - 1);
            }
        }

        let c = out_coord[3];

        data[linear] = input.get_f32(&[b, y, x, c]);

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
