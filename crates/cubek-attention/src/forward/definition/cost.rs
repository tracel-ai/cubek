use cubecl::{
    client::ComputeClient,
    ir::{ElemType, FloatKind, StorageType},
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
    /// Whether a causal mask skips half the score matrix calculations.
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

        let scores = batch * num_heads * seq_q;

        // Causal masking skips roughly half of the score matrix and associated matmuls.
        let causal_div = if self.causal { 2 } else { 1 };

        let qk_ops = scores * seq_kv * (2 * head_dim - 1) / causal_div;
        let sv_ops = scores * val_dim * (2 * seq_kv - 1) / causal_div;

        let elements = |seq: usize, dim: usize| batch * num_heads * seq * dim;
        let mask_bytes = match self.masked {
            true => elements(seq_q, seq_kv) * self.types.mask.size() / causal_div,
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
        let acc = StorageType::Scalar(ElemType::Float(FloatKind::F32));

        let cmma_tile =
            select_cmma_tile(client, self.types.query, self.types.key, acc, UNCONSTRAINED);

        compute_throughput_key(
            cmma_tile,
            self.types.query.elem_type(),
            ElemType::Float(FloatKind::F32),
        )
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
        let f32 = StorageType::Scalar(ElemType::Float(FloatKind::F32));

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
    fn a_causal_mask_halves_the_op_count() {
        let causal = AttentionCost {
            causal: true,
            ..cost()
        };

        assert_eq!(causal.work().compute_ops, cost().work().compute_ops / 2);
    }

    #[test]
    fn a_decode_step_costs_one_pass_over_the_cache() {
        // seq_q = 1 represents decode; causal division must not result in zero compute ops.
        let decode = AttentionCost {
            dims: AttentionDims {
                seq_q: 1,
                ..cost().dims
            },
            causal: true,
            ..cost()
        };

        assert!(decode.work().compute_ops > 0);
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
        // Causal mask bytes are halved along with the visited score blocks.
        let masked = AttentionCost {
            masked: true,
            causal: true,
            ..cost()
        };
        let causal = AttentionCost {
            causal: true,
            ..cost()
        };

        assert_eq!(
            masked.work().bytes,
            causal.work().bytes + 2 * 8 * 16 * 4 / 2
        );
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

