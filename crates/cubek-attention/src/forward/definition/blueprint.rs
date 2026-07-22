use cubek_std::TileSize;
use cubek_std::cube_count::{Count3d, CubeCountPlan};

use crate::forward::definition::{AttentionDims, AttentionVectorSizes, HypercubeBlueprint};

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AttentionBlueprint {
    pub hypercube_blueprint: HypercubeBlueprint,

    pub tiling_scheme: AttentionTilingScheme,
    pub plane_dim: u32,

    pub two_rows_in_array_tile: bool,

    pub vector_sizes: AttentionVectorSizes,

    pub masked: bool,
    pub causal: bool,

    pub check_bounds: AttentionCheckBounds,
}

impl AttentionBlueprint {
    /// Build the [CubeCountPlan] for an attention problem, with 2D conceptual
    /// axes `(seq_q_cubes, batch * num_heads)` (z is unused).
    pub fn cube_count_plan(
        &self,
        dims: &AttentionDims,
        max_cube_count: &(u32, u32, u32),
    ) -> CubeCountPlan {
        let seq_q_cubes = (dims.seq_q as u32).div_ceil(
            self.tiling_scheme.tile_size.seq_q
                * self.tiling_scheme.partition_size.seq_q
                * self.tiling_scheme.stage_size.seq_q,
        );
        let batch_heads = (dims.batch * dims.num_heads) as u32;
        let target_count = Count3d {
            x: seq_q_cubes,
            y: batch_heads,
            z: 1,
        };
        CubeCountPlan::from_blueprint(&self.hypercube_blueprint, target_count, max_cube_count)
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionTilingScheme {
    pub tile_size: AttentionTileSize,
    pub partition_size: AttentionPartitionSize,
    pub stage_size: AttentionStageSize,
}

impl AttentionTilingScheme {
    pub fn elements_in_tile_seq_q(&self) -> u32 {
        self.tile_size.seq_q
    }

    pub fn elements_in_tile_seq_kv(&self) -> u32 {
        self.tile_size.seq_kv
    }

    pub fn elements_in_partition_seq_q(&self) -> u32 {
        self.partition_size.seq_q * self.elements_in_tile_seq_q()
    }

    pub fn elements_in_partition_seq_kv(&self) -> u32 {
        self.partition_size.seq_kv * self.elements_in_tile_seq_kv()
    }

    pub fn elements_in_partition_head_dim(&self) -> u32 {
        self.partition_size.head_dim * self.tile_size.head_dim
    }

    pub fn elements_in_partition_val_dim(&self) -> u32 {
        self.partition_size.val_dim * self.tile_size.val_dim
    }

    pub fn elements_in_stage_seq_q(&self) -> u32 {
        self.stage_size.seq_q * self.elements_in_partition_seq_q()
    }

    pub fn check_bounds(&self, problem: &AttentionDims) -> AttentionCheckBounds {
        AttentionCheckBounds {
            seq_q: !(problem.seq_q as u32).is_multiple_of(self.elements_in_stage_seq_q()),
            seq_kv: !(problem.seq_kv as u32).is_multiple_of(self.elements_in_partition_seq_kv()),
            head_dim: !(problem.head_dim as u32)
                .is_multiple_of(self.elements_in_partition_head_dim()),
            val_dim: !(problem.val_dim as u32).is_multiple_of(self.elements_in_partition_val_dim()),
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
// Score matmul: (seq_q, head_dim) @ (head_dim, seq_kv) → (seq_q, seq_kv)
// Value matmul: (seq_q, seq_kv) @ (seq_kv, val_dim) → (seq_q, val_dim)
pub struct AttentionTileSize {
    pub seq_q: u32,    // Query sequence length (m of both matmuls)
    pub head_dim: u32, // Head/embedding dimension, Shared Q-K dimension (k of score matmul)
    pub seq_kv: u32,   // Key/Value sequence length (n of score matmul, k of value matmul)
    pub val_dim: u32,  // Value output dimension (n of value matmul)
}

impl AttentionTileSize {
    pub fn from_max_vector_sizes(vector_sizes: &AttentionVectorSizes) -> Self {
        fn lcm(a: usize, b: usize) -> usize {
            a / gcd(a, b) * b
        }

        fn gcd(mut a: usize, mut b: usize) -> usize {
            while b != 0 {
                let tmp = b;
                b = a % b;
                a = tmp;
            }
            a
        }

        let head_dim = lcm(vector_sizes.query, vector_sizes.key);
        let val_dim = lcm(vector_sizes.value, vector_sizes.out);
        let seq_kv = lcm(vector_sizes.key, vector_sizes.mask);

        // Independent from vectorization
        let seq_q = 8;

        AttentionTileSize {
            seq_q,
            head_dim: head_dim as u32,
            seq_kv: seq_kv as u32,
            val_dim: val_dim as u32,
        }
    }

    pub fn to_score_matmul_tile_size(&self) -> TileSize {
        TileSize {
            m: self.seq_q,
            n: self.seq_kv,
            k: self.head_dim,
        }
    }

    pub fn to_value_matmul_tile_size(&self) -> TileSize {
        TileSize {
            m: self.seq_q,
            n: self.val_dim,
            k: self.seq_kv,
        }
    }
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionPartitionSize {
    pub seq_q: u32,
    pub head_dim: u32,
    pub seq_kv: u32,
    pub val_dim: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionStageSize {
    // Other dims don't make sense
    pub seq_q: u32,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct AttentionCheckBounds {
    pub seq_q: bool,
    pub seq_kv: bool,
    pub head_dim: bool,
    pub val_dim: bool,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scheme() -> AttentionTilingScheme {
        // Spans: seq_q stage = 4 * 8 * 2 = 64, seq_kv partition = 4 * 8 = 32,
        // head_dim partition = 8 * 2 = 16, val_dim partition = 8 * 2 = 16.
        AttentionTilingScheme {
            tile_size: AttentionTileSize {
                seq_q: 8,
                head_dim: 8,
                seq_kv: 8,
                val_dim: 8,
            },
            partition_size: AttentionPartitionSize {
                seq_q: 4,
                head_dim: 2,
                seq_kv: 4,
                val_dim: 2,
            },
            stage_size: AttentionStageSize { seq_q: 2 },
        }
    }

    fn dims(seq_q: usize, seq_kv: usize, head_dim: usize, val_dim: usize) -> AttentionDims {
        AttentionDims {
            batch: 1,
            num_heads: 1,
            seq_q,
            seq_kv,
            head_dim,
            val_dim,
        }
    }

    #[test]
    fn no_checks_when_spans_divide_problem() {
        let checks = scheme().check_bounds(&dims(128, 64, 32, 16));
        assert!(!checks.seq_q);
        assert!(!checks.seq_kv);
        assert!(!checks.head_dim);
        assert!(!checks.val_dim);
    }

    #[test]
    fn no_checks_when_problem_equals_spans() {
        let checks = scheme().check_bounds(&dims(64, 32, 16, 16));
        assert!(!checks.seq_q);
        assert!(!checks.seq_kv);
        assert!(!checks.head_dim);
        assert!(!checks.val_dim);
    }

    /// The historical inversion: when the problem dim divides the span
    /// (smaller problem than one span), checks were wrongly disabled and the
    /// tail ran unmasked.
    #[test]
    fn checks_when_problem_divides_span() {
        let checks = scheme().check_bounds(&dims(32, 16, 8, 8));
        assert!(checks.seq_q);
        assert!(checks.seq_kv);
        assert!(checks.head_dim);
        assert!(checks.val_dim);
    }

    #[test]
    fn checks_when_problem_coprime_with_span() {
        let checks = scheme().check_bounds(&dims(1500, 1500, 17, 23));
        assert!(checks.seq_q);
        assert!(checks.seq_kv);
        assert!(checks.head_dim);
        assert!(checks.val_dim);
    }
}
