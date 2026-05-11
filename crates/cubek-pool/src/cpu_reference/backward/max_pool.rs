use crate::{cpu_reference::decode_index, definition::MaxPoolOptions};
use cubek_test_utils::HostData;

pub fn run_max_pool_backward<const N: usize>(
    grad_output: &HostData,
    indices: &HostData,
    _opts: &MaxPoolOptions<N>,
    grad_input_dims: &[usize],
    _grad_output_dims: &[usize],
    grad_input_strides: &[usize],
) -> Vec<f32> {
    let total: usize = grad_input_dims.iter().product();
    let mut grad_input = vec![0.0; total];

    let batch_idx = 0;
    let channel_idx = N + 1;

    for i in 0..total {
        let coords = decode_index(i, grad_input_dims, grad_input_strides);
        let batch = coords[batch_idx];
        let channel = coords[channel_idx];
        let spatial_in = &coords[1..N + 1];

        let mut current_spatial_idx = 0;
        let mut multiplier = 1;
        for d in (0..N).rev() {
            current_spatial_idx += spatial_in[d] * multiplier;
            multiplier *= grad_input_dims[d + 1];
        }

        let mut grad_acc = 0.0;

        if N == 2 {
            for out_coords in indices.iter_indices() {
                if out_coords[batch_idx] != batch || out_coords[channel_idx] != channel {
                    continue;
                }

                let max_idx_at_output = indices.get_i32(&out_coords);
                if max_idx_at_output == current_spatial_idx as i32 {
                    grad_acc += grad_output.get_f32(&out_coords);
                }
            }
        }

        grad_input[i] = grad_acc;
    }

    grad_input
}
