use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use super::super::{contiguous_strides, for_each_output_coord};

pub fn reference_nearest_backward(
    out_grad: &HostData,
    output_shape: &[usize],
    _: bool,
    progress: Option<&Progress>,
) -> HostData {
    let (out_h, out_w) = (output_shape[1], output_shape[2]);
    let (grad_h, grad_w) = (out_grad.shape[1], out_grad.shape[2]);
    let mut data = vec![0.0f32; output_shape.iter().product()];

    for_each_output_coord(output_shape, |linear, out_coord| {
        let b = out_coord[0];
        let out_y = out_coord[1];
        let out_x = out_coord[2];
        let c = out_coord[3];

        let grad_y_start = start_index(out_y, grad_h, out_h);
        let grad_y_end = end_index(out_y, grad_h, out_h);
        let grad_x_start = start_index(out_x, grad_w, out_w);
        let grad_x_end = end_index(out_x, grad_w, out_w);

        let mut sum = 0.0f32;
        for grad_y in grad_y_start..grad_y_end {
            for grad_x in grad_x_start..grad_x_end {
                sum += out_grad.get_f32(&[b, grad_y, grad_x, c]);
            }
        }

        data[linear] = sum;

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

fn start_index(input_index: usize, output_size: usize, input_size: usize) -> usize {
    let numerator = (input_index * output_size) as f32;
    let div = (numerator / input_size as f32).ceil();

    div as usize
}

fn end_index(input_index: usize, output_size: usize, input_size: usize) -> usize {
    let numerator = ((input_index + 1) * output_size) as f32;
    let div = (numerator / input_size as f32).ceil() as usize;

    div.min(output_size)
}
