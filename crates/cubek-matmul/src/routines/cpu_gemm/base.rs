//! The CpuGemm routine
//!
//! # Supported layouts
//!
//! Each operand carries its own [`InnerLayout`](crate::definition::InnerLayout), and the
//! kernel reads it through a layout-agnostic view — so the three operands may differ. The
//! supported set (per operand, independently):
//!
//! - **Row-major** (`cols` contiguous) — the only layout that takes the vectorized N path
//!   (rhs *and* output must both be row-major to vectorize; otherwise scalar).
//! - **Col-major** (`rows` contiguous) — correct, scalar.
//! - **Tiled** (nested contiguous blocks, single or recursive, rectangular) — correct,
//!   scalar. Only reachable via the direct [`launch_ref`] (a tiled buffer isn't a plain
//!   strided binding); the [`Strategy`](crate::strategy::Strategy) entry deduces row/col.
//!
//! Lhs, rhs, and the accumulator each carry their own element type; the leaf reads every
//! operand in its native dtype and widens the inputs into the accumulate element, so a
//! mixed-precision GEMM (e.g. `f16` inputs, `f32` accumulate) runs through the same kernel.
//!
//! # Rejected (returns [`MatmulSetupError`])
//!
//! - **Quantized inputs** — unsupported.
//! - **Non-contiguous strided bindings** on the [`Strategy`](crate::strategy::Strategy) path
//!   — a binding contiguous in neither matrix axis is not a plain row/col matrix and is
//!   rejected by the strided deduction

use std::fmt::Display;

use cubecl::Runtime;
use cubek_tile::{Axis, AxisSet, Constraint, Facet, LayoutRequest};

use crate::{
    definition::{MatmulProblem, MatmulSetupError},
    routines::{BlueprintStrategy, DeviceSettings, Routine},
};

// Matmul axes
pub(crate) const M: Axis = Axis(0);
pub(crate) const N: Axis = Axis(1);
pub(crate) const K: Axis = Axis(2);
/// The axis for output batch dimension `i` (outermost is `0`).
pub(crate) fn batch_axis(i: usize) -> Axis {
    Axis(3 + i as u8)
}

/// L1 data-cache budget the blocking targets, in bytes. Conservative constant until
/// the runtime exposes per-core cache sizes.
const L1_BYTES: usize = 32 * 1024;

/// A fully-resolved CpuGemm plan: the cuboid sub-tile each cube computes. `tile_n`
/// rides SIMD lines (N is the vectorized axis), `tile_m` is register rows, `tile_k`
/// is the in-cube contraction depth.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuGemmBlueprint {
    pub tile_m: usize,
    pub tile_n: usize,
    pub tile_k: usize,
}

impl CpuGemmBlueprint {
    /// Reject a degenerate blueprint.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, _problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        if self.tile_m == 0 || self.tile_n == 0 || self.tile_k == 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm blocks must be non-zero, got {}x{}x{}",
                self.tile_m, self.tile_n, self.tile_k
            ))));
        }
        Ok(())
    }
}

/// `alpha` slides the M/N microtile between favouring
/// - parallelism (→0: many small cubes)
/// - reuse (→1: fewer fat cubes with deeper cache residency).
#[derive(Clone, Debug)]
pub struct CpuGemmStrategy {
    pub alpha: f32,
}

impl Default for CpuGemmStrategy {
    fn default() -> Self {
        Self { alpha: 0.5 }
    }
}

impl Display for CpuGemmStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "_a{}", self.alpha)
    }
}

/// The per-operand layout wishes a matmul strategy declares. Burn pairs a relayout with the
/// strategy to satisfy these; the kernel still runs on whatever it is handed.
#[allow(dead_code)]
pub struct MatmulLayoutRequest {
    pub lhs: LayoutRequest,
    pub rhs: LayoutRequest,
    pub out: LayoutRequest,
}

impl CpuGemmStrategy {
    /// CpuGemm vectorizes over `N`, so it wants `N` innermost (contiguous) on `rhs` and the
    /// output. `lhs` is broadcast scalar, so its layout is free. Preferred, not required: the
    /// kernel falls back to scalar when a delivered operand puts another axis innermost.
    #[allow(dead_code)]
    pub fn layout_request() -> MatmulLayoutRequest {
        let n_innermost =
            || LayoutRequest::new().with(Constraint::preferred(Facet::Innermost(AxisSet::one(N))));
        MatmulLayoutRequest {
            lhs: LayoutRequest::new(),
            rhs: n_innermost(),
            out: n_innermost(),
        }
    }
}

/// Pairs the [`CpuGemmStrategy`] knob with the [`CpuGemmBlueprint`] plan.
pub struct CpuGemmRoutine;

impl Routine<()> for CpuGemmRoutine {
    type Strategy = CpuGemmStrategy;
    type Blueprint = CpuGemmBlueprint;
}

impl CpuGemmRoutine {
    /// Resolve `strategy` into a validated cuboid for `problem` on this device.
    #[allow(clippy::result_large_err)]
    pub fn blueprint<R: Runtime>(
        strategy: &BlueprintStrategy<(), CpuGemmRoutine>,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> Result<CpuGemmBlueprint, MatmulSetupError> {
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(strategy) => {
                Self::select(strategy, problem, device_settings)
            }
        };
        blueprint.validate(problem)?;
        Ok(blueprint)
    }

    /// The tile-size heuristic. `alpha` picks the M/N microtile edge between one SIMD
    /// vector (max parallelism) and the largest square C tile that still leaves room in
    /// L1 for the streaming A/B panels (max reuse). A parallelism floor shrinks it
    /// further if cubes would leave cores idle, then `tile_k` fills the remaining cache
    /// depth while the C accumulator stays resident.
    fn select<R: Runtime>(
        strategy: &CpuGemmStrategy,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> CpuGemmBlueprint {
        let (m, n, k, batch) = (problem.m, problem.n, problem.k, problem.num_batches());
        let elem = problem.global_dtypes.out.size().max(1);
        let vw = device_settings.vector_sizes.out.max(1); // SIMD width along N
        let cores = device_settings
            .client
            .properties()
            .hardware
            .num_cpu_cores
            .map(|c| c as usize)
            .unwrap_or(4)
            .max(1);
        let alpha = strategy.alpha.clamp(0.0, 1.0);

        // Microtile edge: lerp between one vector (parallelism) and the largest square C
        // tile that fits half of L1 (reuse), per `alpha`.
        let e_min = vw;
        let e_max = { ((L1_BYTES / (2 * elem)) as f64).sqrt() as usize }.max(e_min);
        let edge = e_min + (alpha * (e_max - e_min) as f32) as usize;

        // N rides SIMD lines: round the microtile edge up to a whole vector, capped at N.
        // The floor is one vector, but never wider than N itself (a narrow `n < vw` has no
        // full line — it rides the scalar path, with `tile_n = n`).
        let n_floor = vw.min(n.max(1));
        let mut tile_n = (edge.div_ceil(vw) * vw).min(n.max(1));
        // M is register rows
        let mut tile_m = edge.clamp(1, m.max(1));

        // Parallelism floor: keep at least one cube per core. Utilisation overrides the
        // `alpha` preference for reuse.
        while batch * m.div_ceil(tile_m) * n.div_ceil(tile_n) < cores
            && (tile_m > 1 || tile_n > n_floor)
        {
            if tile_m >= tile_n {
                tile_m = (tile_m / 2).max(1);
            } else {
                tile_n = (tile_n / 2).max(n_floor);
            }
        }

        // K depth: fill the rest of L1 with the A (tile_m×tile_k) and B (tile_k×tile_n)
        // panels while the C tile (tile_m×tile_n) stays resident.
        let tile_k = ((L1_BYTES / elem).saturating_sub(tile_m * tile_n) / (tile_m + tile_n))
            .clamp(1, k.max(1));

        // Edge tiles are masked, so the heuristic's ideal block stands — just clamp each
        // edge to its axis (a tile no larger than the matrix) and keep it non-zero.
        CpuGemmBlueprint {
            tile_m: tile_m.clamp(1, m.max(1)),
            tile_n: tile_n.clamp(1, n.max(1)),
            tile_k: tile_k.clamp(1, k.max(1)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::definition::InnerLayout;

    #[test]
    fn cpu_gemm_prefers_n_innermost_on_rhs_and_out() {
        let req = CpuGemmStrategy::layout_request();

        // The preferred wish is met exactly when N is contiguous, mirroring the kernel's
        // vectorize-vs-scalar condition on rhs and out.
        assert_eq!(
            req.rhs
                .preference(&InnerLayout::RowMajor.to_concrete([K, N], 16, 16)),
            1
        );
        assert_eq!(
            req.rhs
                .preference(&InnerLayout::ColMajor.to_concrete([K, N], 16, 16)),
            0
        );
        assert_eq!(
            req.out
                .preference(&InnerLayout::RowMajor.to_concrete([M, N], 16, 16)),
            1
        );
        assert_eq!(
            req.out
                .preference(&InnerLayout::ColMajor.to_concrete([M, N], 16, 16)),
            0
        );

        // lhs is broadcast scalar: no layout wish.
        assert!(req.lhs.constraints.is_empty());
    }
}
