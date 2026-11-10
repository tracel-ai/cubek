//! What every attention microkernel computes alone: `softmax(scale · q · kᵀ) · v` over one batch
//! and one head, every key attended, all row-major, and each query's `ln Σ exp(scale · q · kᵀ)`.

use cubecl::zspace::shape;
use cubek_test_utils::{HostData, HostDataVec};

/// An attention on the host: its query, keys and values, small values so the scores stay in a
/// range `f16` holds and the softmax spreads, and what it writes.
pub(crate) struct HostAttention {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
    pub value: Vec<f32>,
    /// The queries and the keys.
    seq: [usize; 2],
    /// The rows of the query and key, and of the value.
    dims: [usize; 2],
}

impl HostAttention {
    /// The inputs of `seq` queries and keys, with rows of `dims` in the query and key and in the
    /// value.
    pub(crate) fn new(seq: [usize; 2], dims: [usize; 2]) -> Self {
        let [queries, keys] = seq;
        let [head, val] = dims;
        let values = |len: usize, modulo: usize, offset: f32| -> Vec<f32> {
            (0..len)
                .map(|i| ((i % modulo) as f32 - offset) / 4.0)
                .collect()
        };
        Self {
            query: values(queries * head, 7, 3.0),
            key: values(keys * head, 5, 2.0),
            value: values(keys * val, 11, 5.0),
            seq,
            dims,
        }
    }

    /// The scale every score is multiplied by: one over the root of the head.
    pub(crate) fn scale(&self) -> f32 {
        1.0 / (self.dims[0] as f32).sqrt()
    }

    /// The output over `[B, KV, G, Q, DV]`, and the logsumexp over `[B, KV, G, Q]`.
    pub(crate) fn out(&self) -> [HostData; 2] {
        let [queries, keys] = self.seq;
        let [head, val] = self.dims;
        let scale = self.scale();
        let mut out = vec![0.0; queries * val];
        let mut logsumexp = vec![0.0; queries];
        for i in 0..queries {
            let scores: Vec<f32> = (0..keys)
                .map(|j| {
                    scale
                        * (0..head)
                            .map(|d| self.query[i * head + d] * self.key[j * head + d])
                            .sum::<f32>()
                })
                .collect();
            let max = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let weights: Vec<f32> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f32 = weights.iter().sum();
            logsumexp[i] = max + total.ln();
            for d in 0..val {
                out[i * val + d] = (0..keys)
                    .map(|j| weights[j] * self.value[j * val + d])
                    .sum::<f32>()
                    / total;
            }
        }
        [
            HostData::row_major(HostDataVec::F32(out), shape![1, 1, 1, queries, val]),
            HostData::row_major(HostDataVec::F32(logsumexp), shape![1, 1, 1, queries]),
        ]
    }
}
