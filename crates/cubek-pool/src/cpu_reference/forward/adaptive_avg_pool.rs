use crate::{
    cpu_reference::{decode_index, forward::decode_index_simple},
    definition::AdaptiveAvgPoolOptions,
};
use cubek_test_utils::HostData;

pub fn run_adaptive_avg_pool<const N: usize>(
    input: &HostData,
    _opts: &AdaptiveAvgPoolOptions<N>,
    out_dims: &[usize],
    in_dims: &[usize],
    out_strides: &[usize],
) -> Vec<f32> {
    let total: usize = out_dims.iter().product();
    let mut out = vec![0.0; total];

    for i in 0..total {
        let coords = decode_index(i, out_dims, out_strides);
        let (batch, channel, spatial_out) = (coords[0], coords[N + 1], &coords[1..N + 1]);

        let mut starts = [0; N];
        let mut ends = [0; N];
        for d in 0..N {
            starts[d] = (spatial_out[d] * in_dims[d + 1]) / out_dims[d + 1];
            ends[d] = ((spatial_out[d] + 1) * in_dims[d + 1])
                .div_ceil(out_dims[d + 1])
                .min(in_dims[d + 1]);
        }

        let k_dims: Vec<usize> = (0..N).map(|d| ends[d] - starts[d]).collect();
        let total_k: usize = k_dims.iter().product();
        let (mut sum, mut count) = (0.0, 0);

        for k_idx in 0..total_k {
            let k_coords = decode_index_simple(k_idx, &k_dims);
            let mut in_coords = vec![0; N + 2];
            in_coords[0] = batch;
            in_coords[N + 1] = channel;
            for d in 0..N {
                in_coords[d + 1] = starts[d] + k_coords[d];
            }

            sum += input.get_f32(&in_coords);
            count += 1;
        }
        out[i] = if count > 0 { sum / count as f32 } else { 0.0 };
    }
    out
}
