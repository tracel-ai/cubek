use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use crate::definition::NearestMode;

use super::super::{contiguous_strides, for_each_output_coord};

pub fn reference_nearest_backward(
    out_grad: &HostData,
    output_shape: &[usize],
    nearest_mode: NearestMode,
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

        let grad_y_start = start_index(out_y, grad_h, out_h, nearest_mode);
        let grad_y_end = end_index(out_y, grad_h, out_h, nearest_mode);
        let grad_x_start = start_index(out_x, grad_w, out_w, nearest_mode);
        let grad_x_end = end_index(out_x, grad_w, out_w, nearest_mode);

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

fn start_index(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    nearest_mode: NearestMode,
) -> usize {
    let ratio = (input_index * output_size) as f32 / input_size as f32;
    match nearest_mode {
        NearestMode::Floor => ratio.ceil() as usize,
        NearestMode::Exact => (ratio - 0.5).ceil().max(0.0) as usize,
    }
}

fn end_index(
    input_index: usize,
    output_size: usize,
    input_size: usize,
    nearest_mode: NearestMode,
) -> usize {
    start_index(input_index + 1, output_size, input_size, nearest_mode).min(output_size)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn floor_and_exact_have_distinct_inverse_boundaries() {
        let floor: Vec<_> = (0..=4)
            .map(|index| start_index(index, 10, 4, NearestMode::Floor).min(10))
            .collect();
        let exact: Vec<_> = (0..=4)
            .map(|index| start_index(index, 10, 4, NearestMode::Exact).min(10))
            .collect();

        assert_eq!(floor, [0, 3, 5, 8, 10]);
        assert_eq!(exact, [0, 2, 5, 7, 10]);
    }
}
