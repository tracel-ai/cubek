use crate::eval::cpu_reference::decode_index_simple;
use cubek_test_utils::HostData;

pub fn run_adaptive_avg_pool_backward<const N: usize>(
    grad_output: &HostData,
    grad_input_dims: &[usize],
    grad_output_dims: &[usize],
    grad_input_strides: &[usize],
) -> Vec<f32> {
    let total: usize = grad_input_dims.iter().product();
    let mut grad_input = vec![0.0; total];
    let batch_size = grad_output_dims[0];
    let channels = grad_output_dims[N + 1];
    let spatial_output = &grad_output_dims[1..N + 1];
    let total_spatial_output: usize = spatial_output.iter().product();

    for batch in 0..batch_size {
        for output_linear in 0..total_spatial_output {
            let output_coords = decode_index_simple(output_linear, spatial_output);
            let mut starts = [0; N];
            let mut ends = [0; N];
            for d in 0..N {
                starts[d] = start_index(
                    output_coords[d],
                    grad_output_dims[d + 1],
                    grad_input_dims[d + 1],
                );
                ends[d] = end_index(
                    output_coords[d],
                    grad_output_dims[d + 1],
                    grad_input_dims[d + 1],
                );
            }

            let window_shape: [usize; N] = core::array::from_fn(|d| ends[d] - starts[d]);
            let window_volume: usize = window_shape.iter().product();

            for channel in 0..channels {
                let mut grad_coords = Vec::with_capacity(N + 2);
                grad_coords.push(batch);
                grad_coords.extend_from_slice(&output_coords);
                grad_coords.push(channel);
                let contribution = grad_output.get_f32(&grad_coords) / window_volume as f32;

                for window_linear in 0..window_volume {
                    let window_coords = decode_index_simple(window_linear, &window_shape);
                    let mut input_offset = batch * grad_input_strides[0];
                    for d in 0..N {
                        input_offset += (starts[d] + window_coords[d]) * grad_input_strides[d + 1];
                    }
                    input_offset += channel * grad_input_strides[N + 1];
                    grad_input[input_offset] += contribution;
                }
            }
        }
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
