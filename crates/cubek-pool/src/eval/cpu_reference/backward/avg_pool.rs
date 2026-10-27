use crate::{definition::AvgPoolOptions, eval::cpu_reference::decode_index};
use cubek_test_utils::HostData;

pub fn run_avg_pool_backward<const N: usize>(
    grad_output: &HostData,
    opts: &AvgPoolOptions<N>,
    grad_input_dims: &[usize],
    grad_output_dims: &[usize],
    grad_input_strides: &[usize],
) -> Vec<f32> {
    let total: usize = grad_input_dims.iter().product();
    let mut grad_input = vec![0.0; total];

    if N != 2 {
        return grad_input;
    }

    let stride_h = opts.window.stride[0] as i32;
    let stride_w = opts.window.stride[1] as i32;
    let pad_h = opts.window.padding[0] as i32;
    let pad_w = opts.window.padding[1] as i32;
    let kernel_h = opts.window.kernel_size[0] as i32;
    let kernel_w = opts.window.kernel_size[1] as i32;
    let dilation_h = opts.dilation[0] as i32;
    let dilation_w = opts.dilation[1] as i32;
    let extent_h = (kernel_h - 1) * dilation_h + 1;
    let extent_w = (kernel_w - 1) * dilation_w + 1;

    let grad_h = grad_output_dims[1] as i32;
    let grad_w = grad_output_dims[2] as i32;

    for (i, grad_val) in grad_input.iter_mut().enumerate().take(total) {
        let coords = decode_index(i, grad_input_dims, grad_input_strides);
        let batch = coords[0] as i32;
        let ih = coords[1] as i32;
        let iw = coords[2] as i32;
        let channel = coords[3] as i32;

        let kms_h = extent_h - stride_h;
        let kms_w = extent_w - stride_w;

        let oh_start = ((ih + pad_h - kms_h).max(0)) / stride_h;
        let ow_start = ((iw + pad_w - kms_w).max(0)) / stride_w;
        let oh_end = ((ih + pad_h) / stride_h + 1).min(grad_h);
        let ow_end = ((iw + pad_w) / stride_w + 1).min(grad_w);

        let begin_h = ih + pad_h;
        let begin_w = iw + pad_w;

        let out_h = grad_input_dims[1] as i32;
        let out_w = grad_input_dims[2] as i32;
        let border_bottom = out_h + pad_h;
        let border_right = out_w + pad_w;

        let mut grad_acc = 0.0f32;

        for oh in oh_start..oh_end {
            let ih_start = oh * stride_h;
            let ih_end = ih_start + extent_h;

            if begin_h >= ih_start && begin_h < ih_end && (begin_h - ih_start) % dilation_h == 0 {
                for ow in ow_start..ow_end {
                    let iw_start = ow * stride_w;
                    let iw_end = iw_start + extent_w;

                    if begin_w >= iw_start
                        && begin_w < iw_end
                        && (begin_w - iw_start) % dilation_w == 0
                    {
                        let out_coords =
                            vec![batch as usize, oh as usize, ow as usize, channel as usize];
                        let grad_val = grad_output.get_f32(&out_coords);

                        if opts.count_include_pad {
                            let padded_h = (0..kernel_h)
                                .filter(|k| ih_start + k * dilation_h < out_h + 2 * pad_h)
                                .count();
                            let padded_w = (0..kernel_w)
                                .filter(|k| iw_start + k * dilation_w < out_w + 2 * pad_w)
                                .count();
                            grad_acc += grad_val / (padded_h * padded_w) as f32;
                        } else {
                            let ih_diff = (0..kernel_h)
                                .filter(|k| {
                                    let pos = ih_start + k * dilation_h;
                                    pos >= pad_h && pos < border_bottom
                                })
                                .count() as f32;
                            let iw_diff = (0..kernel_w)
                                .filter(|k| {
                                    let pos = iw_start + k * dilation_w;
                                    pos >= pad_w && pos < border_right
                                })
                                .count() as f32;
                            grad_acc += grad_val / (ih_diff * iw_diff);
                        }
                    }
                }
            }
        }

        *grad_val = grad_acc;
    }

    grad_input
}
