use cubecl::zspace::Shape;
use cubek_test_utils::{HostData, HostDataVec, Progress};

use super::{contiguous_strides, max_rank};

/// TopK returns the `k` largest values per output slice.
/// The output shape has `axis` set to `k` (rather than `1` as for scalar reductions).
pub fn reference_topk(
    input: &HostData,
    axis: usize,
    k: usize,
    progress: Option<&Progress>,
) -> HostData {
    let axis_len = input.shape[axis];

    let mut out_shape_vec: Vec<usize> = input.shape.iter().copied().collect();
    out_shape_vec[axis] = k;

    let num_outputs_batches: usize = input.shape.iter().product::<usize>() / axis_len;
    let mut data = vec![0.0f32; num_outputs_batches * k];

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

        let mut candidates: Vec<(f32, u32)> = Vec::with_capacity(axis_len);
        let mut coord = batch_coord.clone();
        for i in 0..axis_len {
            coord[axis] = i;
            candidates.push((input.get_f32(&coord), i as u32));
        }
        candidates.sort_by(|a, b| max_rank(*a, *b));

        for i in 0..k {
            coord[axis] = i;
            let idx = coord
                .iter()
                .zip(out_strides.iter())
                .map(|(c, s)| c * s)
                .sum::<usize>();
            data[idx] = if i < candidates.len() {
                candidates[i].0
            } else {
                f32::MIN
            };
            if let Some(p) = progress {
                p.bump();
            }
        }
    }

    HostData {
        data: HostDataVec::F32(data),
        shape: Shape::from(out_shape_vec),
        strides: out_strides,
    }
}
