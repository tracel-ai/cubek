use crate::{
    cpu_reference::{decode_index, forward::decode_index_simple},
    definition::AvgPoolOptions,
};
use cubek_test_utils::HostData;

pub fn run_avg_pool<const N: usize>(
    input: &HostData,
    opts: &AvgPoolOptions<N>,
    out_dims: &[usize],
    in_dims: &[usize],
    out_strides: &[usize],
) -> Vec<f32> {
    let total: usize = out_dims.iter().product();
    let mut out = vec![0.0; total];
    let kernel_dims = opts.window.kernel_size;
    let total_k_elems: usize = kernel_dims.iter().product();

    let padded_dims: Vec<usize> = (0..N)
        .map(|d| in_dims[d + 1] + 2 * opts.window.padding[d])
        .collect();

    for i in 0..total {
        let coords = decode_index(i, out_dims, out_strides);
        let (batch, channel, spatial_out) = (coords[0], coords[N + 1], &coords[1..N + 1]);
        let (mut sum, mut count) = (0.0f32, 0u32);

        for k_idx in 0..total_k_elems {
            let k_coords = decode_index_simple(k_idx, &kernel_dims);
            let mut is_in_input = true;
            let mut is_in_padded = true;

            let mut in_coords = vec![0; N + 2];
            in_coords[0] = batch;
            in_coords[N + 1] = channel;

            for d in 0..N {
                let id_pad = spatial_out[d] * opts.window.stride[d] + k_coords[d];
                let id_signed = id_pad as isize - opts.window.padding[d] as isize;

                if id_signed < 0 || id_signed >= in_dims[d + 1] as isize {
                    is_in_input = false;
                } else {
                    in_coords[d + 1] = id_signed as usize;
                }
                if id_pad >= padded_dims[d] {
                    is_in_padded = false;
                }
            }

            if is_in_input {
                sum += input.get_f32(&in_coords);
                if !opts.count_include_pad {
                    count += 1;
                }
            }
            if opts.count_include_pad && is_in_padded {
                count += 1;
            }
        }
        out[i] = if count > 0 { sum / count as f32 } else { 0.0 };
    }
    out
}
