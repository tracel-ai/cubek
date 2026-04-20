use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec};

use super::contiguous_strides;

/// ArgTopK returns the `k` axis indices of the top values per output slice.
/// The output shape has `axis` set to `k` (rather than `1` as for scalar reductions).
pub fn reference_argtopk(input: &HostData, axis: usize, k: u32) -> HostData {
    let axis_len = input.shape[axis];
    let k_usize = k as usize;

    let mut out_shape_vec: Vec<usize> = input.shape.iter().copied().collect();
    out_shape_vec[axis] = k_usize;

    let num_outputs_batches: usize = input.shape.iter().product::<usize>() / axis_len;
    let mut data = vec![0.0f32; num_outputs_batches * k_usize];

    let out_strides = contiguous_strides(&out_shape_vec);

    let rank = input.shape.len();
    let mut batch_coord = vec![0usize; rank];

    let mut batch_shape: Vec<usize> = input.shape.iter().copied().collect();
    batch_shape[axis] = 1;

    for batch in 0..num_outputs_batches {
        let mut rem = batch;
        for d in (0..rank).rev() {
            batch_coord[d] = rem % batch_shape[d];
            rem /= batch_shape[d];
        }

        let mut pairs: Vec<(f32, u32)> = Vec::with_capacity(axis_len);
        let mut coord = batch_coord.clone();
        for i in 0..axis_len {
            coord[axis] = i;
            pairs.push((input.get_f32(&coord), i as u32));
        }
        pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        for i in 0..k_usize {
            coord[axis] = i;
            let idx = coord
                .iter()
                .zip(out_strides.iter())
                .map(|(c, s)| c * s)
                .sum::<usize>();
            data[idx] = if i < pairs.len() {
                pairs[i].1 as f32
            } else {
                u32::MAX as f32
            };
        }
    }

    HostData {
        data: HostDataVec::F32(data),
        shape: Shape::from(out_shape_vec),
        strides: out_strides,
    }
}
