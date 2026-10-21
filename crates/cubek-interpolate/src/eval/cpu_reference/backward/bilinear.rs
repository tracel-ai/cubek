use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use super::super::{contiguous_strides, for_each_output_coord};

/// Scatter each output gradient onto the four taps the forward pass sampled.
///
/// This is the forward reference's adjoint, written as a scatter on purpose:
/// the kernel gathers per input pixel, so the two only agree when the kernel's
/// inverted support intervals are right.
pub fn reference_bilinear_backward(
    out_grad: &HostData,
    output_shape: &[usize],
    align_corners: bool,
    progress: Option<&Progress>,
) -> HostData {
    let (in_h, in_w) = (output_shape[1], output_shape[2]);
    let (grad_h, grad_w) = (out_grad.shape[1], out_grad.shape[2]);
    let strides = contiguous_strides(output_shape);
    let mut data = vec![0.0f64; output_shape.iter().product()];

    for_each_output_coord(&out_grad.shape, |_, grad_coord| {
        let [b, y, x, c] = [grad_coord[0], grad_coord[1], grad_coord[2], grad_coord[3]];
        let grad = out_grad.get_f32(&[b, y, x, c]) as f64;
        for (iy, wy) in taps(y, in_h, grad_h, align_corners) {
            for (ix, wx) in taps(x, in_w, grad_w, align_corners) {
                let index = b * strides[0] + iy * strides[1] + ix * strides[2] + c * strides[3];
                data[index] += grad * wy * wx;
            }
        }

        if let Some(p) = progress {
            p.bump();
        }
    });

    HostData {
        data: HostDataVec::F32(data.into_iter().map(|v| v as f32).collect()),
        shape: Shape::from(output_shape.to_vec()),
        strides: contiguous_strides(output_shape),
    }
}

/// The two input indices an output index samples along one axis, with weights.
fn taps(
    output: usize,
    input_size: usize,
    output_size: usize,
    align_corners: bool,
) -> [(usize, f64); 2] {
    let position = if align_corners {
        output as f64 * (input_size - 1) as f64 / (output_size - 1).max(1) as f64
    } else {
        ((output as f64 + 0.5) * input_size as f64 / output_size as f64 - 0.5)
            .clamp(0.0, (input_size - 1) as f64)
    };
    let lower = position.floor() as usize;
    let upper = (lower + 1).min(input_size - 1);
    let fraction = position - lower as f64;
    [(lower, 1.0 - fraction), (upper, fraction)]
}
