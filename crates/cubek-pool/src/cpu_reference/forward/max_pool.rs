use crate::{
    cpu_reference::forward::{decode_index, decode_index_simple, get_window_coords},
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
