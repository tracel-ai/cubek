//! What every split-keys merge computes alone: each query's slots, left by workers that each
//! walked a part of the keys, merged into its output and logsumexp. With `M` the slots' largest
//! max and `w_t = exp(max_t − M)`, the output is `Σ w_t · out_t / Σ w_t · sum_t` and the logsumexp
//! `M + ln Σ w_t · sum_t`; a query whose every slot saw nothing writes zeros and the minimum value.

use cubecl::ir::{ElemType, FloatKind};
use cubecl::zspace::shape;
use cubek_test_utils::{HostData, HostDataVec};

pub(crate) const F16: ElemType = ElemType::Float(FloatKind::F16);
pub(crate) const F32: ElemType = ElemType::Float(FloatKind::F32);

/// Slots on the host: every query's, the last query's empty, and what they merge to.
pub(crate) struct HostSlots {
    pub out: Vec<f32>,
    pub max: Vec<f32>,
    pub sum: Vec<f32>,
    queries: usize,
    splits: usize,
    val_dim: usize,
}

impl HostSlots {
    /// The slots of `queries` queries, `splits` each, `val_dim` wide. One slot of the first query
    /// saw nothing; every slot of the last did.
    pub(crate) fn new(queries: usize, splits: usize, val_dim: usize) -> Self {
        let slots = queries * splits;
        let mut max: Vec<f32> = (0..slots)
            .map(|i| ((i * 7) % 11) as f32 / 4.0 - 1.0)
            .collect();
        let mut sum: Vec<f32> = (0..slots)
            .map(|i| ((i * 5) % 9) as f32 / 2.0 + 1.0)
            .collect();
        let out = (0..slots * val_dim)
            .map(|i| ((i * 13) % 17) as f32 / 8.0 - 1.0)
            .collect();
        let empty = |slot: usize, max: &mut Vec<f32>, sum: &mut Vec<f32>| {
            max[slot] = f32::MIN;
            sum[slot] = 0.0;
        };
        empty(splits - 1, &mut max, &mut sum);
        for t in 0..splits {
            empty((queries - 1) * splits + t, &mut max, &mut sum);
        }
        Self {
            out,
            max,
            sum,
            queries,
            splits,
            val_dim,
        }
    }

    /// The merged output `[1, 1, 1, queries, val_dim]`, then the logsumexp `[1, 1, 1, queries]`.
    pub(crate) fn merged(&self) -> [HostData; 2] {
        let (splits, val_dim) = (self.splits, self.val_dim);
        let mut out = Vec::with_capacity(self.queries * val_dim);
        let mut logsumexp = Vec::with_capacity(self.queries);
        for q in 0..self.queries {
            let slots = q * splits..(q + 1) * splits;
            let merged = self.max[slots.clone()]
                .iter()
                .copied()
                .fold(f32::MIN, f32::max);
            let weights: Vec<f32> = slots
                .clone()
                .map(|t| (self.max[t] - merged).exp())
                .collect();
            let total: f32 = slots
                .clone()
                .zip(&weights)
                .map(|(t, w)| self.sum[t] * w)
                .sum();
            logsumexp.push(match total > 0.0 {
                true => merged + total.ln(),
                false => f32::MIN,
            });
            for d in 0..val_dim {
                let value: f32 = slots
                    .clone()
                    .zip(&weights)
                    .map(|(t, w)| self.out[t * val_dim + d] * w)
                    .sum();
                out.push(match total > 0.0 {
                    true => value / total,
                    false => 0.0,
                });
            }
        }
        [
            HostData::row_major(
                HostDataVec::F32(out),
                shape![1, 1, 1, self.queries, val_dim],
            ),
            HostData::row_major(HostDataVec::F32(logsumexp), shape![1, 1, 1, self.queries]),
        ]
    }
}
