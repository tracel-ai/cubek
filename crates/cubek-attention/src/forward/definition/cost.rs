use cubecl::{
    client::ComputeClient,
    ir::{ElemType, FloatKind},
    prelude::Runtime,
    throughput::{ThroughputKey, compute_throughput_key, select_cmma_tile},
    tune::Work,
};

use crate::forward::definition::{AttentionDims, AttentionGlobalTypes, AttentionProblem};

/// Minimal representation of attention cost dependencies, including matrix extents,
/// masking requirements, and element types.
#[derive(Debug, Clone)]
pub struct AttentionCost {
    /// Extents of the query, key, and value tensors.
    pub dims: AttentionDims,
    /// Whether a mask tensor is read.
    pub masked: bool,
    /// Whether a causal mask skips the score matrix above its bottom-right diagonal.
    pub causal: bool,
    /// Global element types of the operands.
    pub types: AttentionGlobalTypes,
}

impl AttentionCost {
    /// Calculates the compute operations and compulsory memory traffic for the attention pass.
    ///
    /// Includes both matmuls (`Q@K^T` and `S@V`) and memory traffic for compulsory input/output operands.
    pub fn work(&self) -> Work {
        let AttentionDims {
            batch,
            num_heads,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        } = self.dims;

        let batch_heads = batch * num_heads;

        // Causal masking drops a score when `j + seq_q > i + seq_kv`, which aligns the
        // diagonal on the bottom right: query `i` visits `i + seq_kv - seq_q + 1` keys,
        // the full rectangle minus the triangle above the diagonal. This is not a flat
        // halving: a decode step (`seq_q = 1`) still visits the whole cache.
        let diagonal = seq_q.min(seq_kv);
        let visited = match self.causal {
            true => diagonal * seq_kv - diagonal * diagonal.saturating_sub(1) / 2,
            false => seq_q * seq_kv,
        };
        // Rows above the diagonal visit nothing, so they contract over nothing either.
        let rows = match self.causal {
            true => diagonal,
            false => seq_q,
        };

        // `2n - 1` ops per output element: n multiplies and n - 1 adds. Saturating so a
        // degenerate extent (an empty KV cache) yields zero instead of underflowing.
        let qk_ops = batch_heads * visited * (2 * head_dim).saturating_sub(1);
        // Every row contracts over the keys it visited, so summing `2 * visited_i - 1`
        // over the rows that visited any gives the adds and multiplies per output column.
        let sv_ops = batch_heads * val_dim * (2 * visited).saturating_sub(rows);

        let elements = |seq: usize, dim: usize| batch_heads * seq * dim;
        // Only the mask entries the kernel actually visits are read.
        let mask_bytes = match self.masked {
            true => batch_heads * visited * self.types.mask.size(),
            false => 0,
        };

        Work {
            compute_ops: qk_ops + sv_ops,
            // Exclude attention bias as fast paths do not read one.
            bytes: elements(seq_q, head_dim) * self.types.query.size()
                + elements(seq_kv, head_dim) * self.types.key.size()
                + elements(seq_kv, val_dim) * self.types.value.size()
                + mask_bytes
                + elements(seq_q, val_dim) * self.types.out.size(),
        }
    }

    /// Generates a throughput key for compute probes representing the peak instruction throughput.
    ///
    /// Uses an unconstrained tile selection to accurately model peak hardware throughput
    /// for attention matmuls.
    pub fn compute_key<R: Runtime>(&self, client: &ComputeClient<R>) -> ThroughputKey {
        const UNCONSTRAINED: (usize, usize, usize) = (usize::MAX, usize::MAX, usize::MAX);

        // Softmax statistics accumulate in f32 regardless of operand types.
        let acc = ElemType::Float(FloatKind::F32);

        let cmma_tile =
            select_cmma_tile(client, self.types.query, self.types.key, acc, UNCONSTRAINED);

        compute_throughput_key(cmma_tile, self.types.query, ElemType::Float(FloatKind::F32))
    }
}

impl From<&AttentionProblem> for AttentionCost {
    fn from(problem: &AttentionProblem) -> Self {
        Self {
            dims: problem.dims.clone(),
            masked: problem.masked,
            causal: problem.options.causal,
            types: problem.global_dtypes.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn f32_types() -> AttentionGlobalTypes {
        let f32 = ElemType::Float(FloatKind::F32);

        AttentionGlobalTypes {
            query: f32,
            key: f32,
            value: f32,
            mask: f32,
            out: f32,
        }
    }

    fn cost() -> AttentionCost {
        AttentionCost {
            dims: AttentionDims {
                batch: 2,
                num_heads: 1,
                seq_q: 8,
                seq_kv: 16,
                head_dim: 4,
                val_dim: 4,
            },
            masked: false,
            causal: false,
            types: f32_types(),
        }
    }

    #[test]
    fn counts_both_matmuls() {
        // Q@K^T ops + S@V ops
        let qk = 2 * 8 * 16 * 7;
        let sv = 2 * 8 * 4 * 31;

        assert_eq!(cost().work().compute_ops, qk + sv);
    }

    #[test]
    fn a_causal_mask_skips_the_scores_above_the_diagonal() {
        // 8 queries over 16 keys, bottom-right aligned: 8 * 16 minus the 8 * 7 / 2
        // triangle the mask drops, so 100 of the 128 score entries are visited.
        let causal = AttentionCost {
            causal: true,
            ..cost()
        };

        let qk = 2 * 100 * 7;
        // 8 rows contract over what they visited: 2 * 100 - 8 ops per val_dim column.
        let sv = 2 * 4 * (2 * 100 - 8);

        assert_eq!(causal.work().compute_ops, qk + sv);
    }

    #[test]
    fn a_decode_step_costs_a_full_pass_over_the_cache() {
        // seq_q = 1 is decode: the single query sits on the diagonal and attends the
        // whole cache, so a causal mask discounts nothing.
        let decode = |causal| AttentionCost {
            dims: AttentionDims {
                seq_q: 1,
                ..cost().dims
            },
            causal,
            ..cost()
        };

        assert_eq!(
            decode(true).work().compute_ops,
            decode(false).work().compute_ops
        );
    }

    #[test]
    fn an_empty_kv_cache_costs_nothing() {
        // Degenerate extents must saturate rather than underflow the op counts.
        let empty = |causal| AttentionCost {
            dims: AttentionDims {
                seq_kv: 0,
                ..cost().dims
            },
            causal,
            masked: true,
            ..cost()
        };

        assert_eq!(empty(false).work().compute_ops, 0);
        assert_eq!(empty(true).work().compute_ops, 0);
    }

    #[test]
    fn counts_the_operands_and_the_output() {
        // (q + k + v + out) elements per batch head * 2 heads * 4 bytes
        assert_eq!(cost().work().bytes, 2 * (32 + 64 + 64 + 32) * 4);
    }

    #[test]
    fn a_mask_is_counted_as_bytes_read() {
        let masked = AttentionCost {
            masked: true,
            ..cost()
        };

        assert_eq!(masked.work().bytes, cost().work().bytes + 2 * 8 * 16 * 4);
    }

    #[test]
    fn a_causal_run_only_reads_the_mask_it_visits() {
        // Only the 100 visited score entries of the 8 x 16 mask are read.
        let masked = AttentionCost {
            masked: true,
            causal: true,
            ..cost()
        };
        let causal = AttentionCost {
            causal: true,
            ..cost()
        };

        assert_eq!(masked.work().bytes, causal.work().bytes + 2 * 100 * 4);
    }

    #[test]
    fn a_causal_mask_does_not_change_the_operand_bytes() {
        // Q, K, and V tensors are fully read even with causal masking.
        let causal = AttentionCost {
            causal: true,
            ..cost()
        };

        assert_eq!(causal.work().bytes, cost().work().bytes);
    }
}
