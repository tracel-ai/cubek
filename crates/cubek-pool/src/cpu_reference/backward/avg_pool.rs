use crate::{cpu_reference::decode_index, definition::AvgPoolOptions};
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

    let grad_h = grad_output_dims[1] as i32;
    let grad_w = grad_output_dims[2] as i32;

    for i in 0..total {
        let coords = decode_index(i, grad_input_dims, grad_input_strides);
        let batch = coords[0] as i32;
        let ih = coords[1] as i32;
        let iw = coords[2] as i32;
        let channel = coords[3] as i32;

        let kms_h = kernel_h - stride_h;
        let kms_w = kernel_w - stride_w;

        let oh_start = ((ih + pad_h - kms_h).max(0)) / stride_h;
        let ow_start = ((iw + pad_w - kms_w).max(0)) / stride_w;
        let oh_end = ((kms_h.max(0) + oh_start).min(grad_h - 1)) + 1;
        let ow_end = ((kms_w.max(0) + ow_start).min(grad_w - 1)) + 1;

        let begin_h = ih + pad_h;
        let begin_w = iw + pad_w;

        let out_h = grad_input_dims[1] as i32;
        let out_w = grad_input_dims[2] as i32;
        let border_bottom = out_h + pad_h;
        let border_right = out_w + pad_w;

        let mut grad_acc = 0.0f32;

        for oh in oh_start..oh_end {
            let ih_start = (oh * stride_h).min(border_bottom).max(pad_h);
            let ih_end = (oh * stride_h + kernel_h).min(border_bottom);

            if begin_h >= ih_start && ih < ih_end {
                for ow in ow_start..ow_end {
                    let iw_start = (ow * stride_w).min(border_right).max(pad_w);
                    let iw_end = (ow * stride_w + kernel_w).min(border_right);

                    if begin_w >= iw_start && iw < iw_end {
                        let out_coords =
                            vec![batch as usize, oh as usize, ow as usize, channel as usize];
                        let grad_val = grad_output.get_f32(&out_coords);

                        if opts.count_include_pad {
                            grad_acc += grad_val / (kernel_h * kernel_w) as f32;
                        } else {
                            let ih_diff = (ih_end - ih_start) as f32;
                            let iw_diff = (iw_end - iw_start) as f32;
                            grad_acc += grad_val / (ih_diff * iw_diff);
                        }
                    }
                }
            }
        }

        grad_input[i] = grad_acc;
    }

    grad_input
}
