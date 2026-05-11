use crate::{cpu_reference::decode_index, definition::AdaptiveAvgPoolOptions};
use cubek_test_utils::HostData;

pub fn run_adaptive_avg_pool_backward<const N: usize>(
    grad_output: &HostData,
    _opts: &AdaptiveAvgPoolOptions<N>,
    grad_input_dims: &[usize],
    grad_output_dims: &[usize],
    grad_input_strides: &[usize],
) -> Vec<f32> {
    let total: usize = grad_input_dims.iter().product();
    let mut grad_input = vec![0.0; total];

    if N != 2 {
        return grad_input;
    }

    let out_h = grad_input_dims[1];
    let out_w = grad_input_dims[2];
    let grad_h = grad_output_dims[1];
    let grad_w = grad_output_dims[2];

    for i in 0..total {
        let coords = decode_index(i, grad_input_dims, grad_input_strides);
        let batch = coords[0];
        let ih = coords[1];
        let iw = coords[2];
        let channel = coords[3];

        let oh_start = start_index(ih, out_h, grad_h);
        let oh_end = end_index(ih, out_h, grad_h);
        let ow_start = start_index(iw, out_w, grad_w);
        let ow_end = end_index(iw, out_w, grad_w);

        let mut grad_acc = 0.0f32;

        for oh in oh_start..oh_end {
            let ih_start = start_index(oh, grad_h, out_h);
            let ih_end = end_index(oh, grad_h, out_h);

            if ih >= ih_start && ih < ih_end {
                for ow in ow_start..ow_end {
                    let iw_start = start_index(ow, grad_w, out_w);
                    let iw_end = end_index(ow, grad_w, out_w);

                    if iw >= iw_start && iw < iw_end {
                        let count = (ih_end - ih_start) * (iw_end - iw_start);
                        let out_coords = vec![batch, oh, ow, channel];
                        grad_acc += grad_output.get_f32(&out_coords) / count as f32;
                    }
                }
            }
        }

        grad_input[i] = grad_acc;
    }

    grad_input
}

fn start_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    (output_size_index * input_size) / output_size
}

fn end_index(output_size_index: usize, output_size: usize, input_size: usize) -> usize {
    let index = (output_size_index + 1) * input_size;
    let index = index.div_ceil(output_size);

    if input_size < index {
        input_size
    } else {
        index
    }
}
