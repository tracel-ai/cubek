use crate::{
    cpu_reference::{
        decode_index,
        forward::{decode_index_simple, get_window_coords},
    },
    definition::MaxPoolOptions,
};
use cubek_test_utils::HostData;

pub fn run_max_pool<const N: usize>(
    input: &HostData,
    opts: &MaxPoolOptions<N>,
    out_dims: &[usize],
    in_dims: &[usize],
    out_strides: &[usize],
) -> Vec<f32> {
    let total: usize = out_dims.iter().product();
    let mut out = vec![0.0; total];
    let kernel_dims = opts.window.kernel_size;
    let total_k_elems: usize = kernel_dims.iter().product();

    for i in 0..total {
        let coords = decode_index(i, out_dims, out_strides);
        let (batch, channel, spatial_out) = (coords[0], coords[N + 1], &coords[1..N + 1]);
        let mut max_val = f32::NEG_INFINITY;

        for k_idx in 0..total_k_elems {
            let k_coords = decode_index_simple(k_idx, &kernel_dims);
            let mut in_coords = vec![0; N + 2];
            in_coords[0] = batch;
            in_coords[N + 1] = channel;

            if let Some(valid_coords) = get_window_coords(
                spatial_out,
                &k_coords,
                opts.window.stride,
                opts.window.padding,
                opts.dilation,
                in_dims,
                in_coords,
            ) {
                max_val = f32::max(max_val, input.get_f32(&valid_coords));
            }
        }
        out[i] = if max_val == f32::NEG_INFINITY {
            0.0
        } else {
            max_val
        };
    }
    out
}

pub fn run_max_pool_with_indices<const N: usize>(
    input: &HostData,
    opts: &MaxPoolOptions<N>,
    out_dims: &[usize],
    in_dims: &[usize],
    out_strides: &[usize],
) -> (Vec<f32>, Vec<i32>) {
    let total: usize = out_dims.iter().product();
    let mut out = vec![0.0; total];
    let mut indices = vec![0i32; total];
    let kernel_dims = opts.window.kernel_size;
    let total_k_elems: usize = kernel_dims.iter().product();

    for i in 0..total {
        let coords = decode_index(i, out_dims, out_strides);
        let (batch, channel, spatial_out) = (coords[0], coords[N + 1], &coords[1..N + 1]);
        let mut max_val = f32::NEG_INFINITY;
        let mut max_idx = 0i32;

        for k_idx in 0..total_k_elems {
            let k_coords = decode_index_simple(k_idx, &kernel_dims);
            let mut in_coords = vec![0; N + 2];
            in_coords[0] = batch;
            in_coords[N + 1] = channel;

            if let Some(valid_coords) = get_window_coords(
                spatial_out,
                &k_coords,
                opts.window.stride,
                opts.window.padding,
                opts.dilation,
                in_dims,
                in_coords,
            ) {
                let val = input.get_f32(&valid_coords);
                if val > max_val {
                    max_val = val;

                    let spatial_in = &valid_coords[1..N + 1];
                    let mut current_spatial_idx = 0usize;
                    let mut multiplier = 1usize;
                    for d in (0..N).rev() {
                        current_spatial_idx += spatial_in[d] * multiplier;
                        multiplier *= in_dims[d + 1];
                    }

                    max_idx = current_spatial_idx as i32;
                }
            }
        }

        out[i] = if max_val == f32::NEG_INFINITY {
            0.0
        } else {
            max_val
        };
        indices[i] = max_idx;
    }

    (out, indices)
}
