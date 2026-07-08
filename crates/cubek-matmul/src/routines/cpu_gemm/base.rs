//! The CpuGemm routine
//!
//! # Supported layouts
//!
//! Each operand carries its own physical layout (row/col-major rides in its strides, storage-tiling
//! depth in its `levels`), and the kernel reads it through a layout-agnostic view — so the three
//! operands may differ. The supported set (per operand, independently):
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
use cubek_tile::Axis;

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

/// Byte budget for the leaf's `tile_m × tile_n` accumulator block, which must stay in vector
/// registers across the `K` loop; overflowing spills every accumulator to the stack (~2×).
/// ~24 of 32 NEON q-registers (~12 of 16 AVX2 ymm), leaving room for the A/B operands.
const ACC_REG_BYTES: usize = 384;

/// The largest divisor of `k` not exceeding `cap` (≥1).
fn divisor_at_most(k: usize, cap: usize) -> usize {
    let cap = cap.clamp(1, k.max(1));
    let mut best = 1;
    for d in 1..=cap {
        if k.is_multiple_of(d) {
            best = d;
        }
    }
    best
}

/// The divisor of `g` nearest `target`, ties going to the larger
fn nearest_divisor(g: usize, target: usize) -> usize {
    let target = target.clamp(1, g.max(1));
    let mut best: usize = 1;
    for d in 1..=g {
        if g.is_multiple_of(d)
            && (d.abs_diff(target) < best.abs_diff(target)
                || (d.abs_diff(target) == best.abs_diff(target) && d > best))
        {
            best = d;
        }
    }
    best
}

/// The `m × n × k` extent of the innermost instruction
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Instruction {
    pub m: usize,
    pub n: usize,
    pub k: usize,
}

/// How many planes a cube's stage tile is divided into, along `m` and `n`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PlaneGrid {
    pub m: usize,
    pub n: usize,
}

/// A fully-resolved CpuGemm plan: the leaf each plane computes ([`Instruction`]) and how
/// finely a cube's stage tile is split across planes ([`PlaneGrid`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CpuGemmBlueprint {
    pub instruction: Instruction,
    pub planes: PlaneGrid,
}

impl CpuGemmBlueprint {
    /// Reject a degenerate blueprint.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, _problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        let (i, p) = (self.instruction, self.planes);
        if i.m == 0 || i.n == 0 || i.k == 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm instruction must be non-zero, got {}x{}x{}",
                i.m, i.n, i.k
            ))));
        }
        if p.m == 0 || p.n == 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm plane grid must be non-zero, got {}x{}",
                p.m, p.n
            ))));
        }
        Ok(())
    }
}

/// `alpha` sets the contraction depth `tile_k` (the leaf is fixed to the register-block
/// size), trading
/// - shallow K, lighter L1 footprint (→0)
/// - deeper K panels, more A/B reuse per accumulator load (→1).
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

    /// The tile-size heuristic. The leaf's accumulator block is sized to the register file
    /// ([`ACC_REG_BYTES`]), not L1; `alpha` sets `k` depth; `cores` becomes the [`PlaneGrid`].
    fn select<R: Runtime>(
        strategy: &CpuGemmStrategy,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> CpuGemmBlueprint {
        let (m, n, k) = (problem.m, problem.n, problem.k);
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

        // Balance the register grid, not the element block: N is read in `vw`-wide lines, so
        // an element-square tile would collapse to one N-line (nr=1) for wide `vw`, starving
        // N-reuse. Aim `tile_n ≈ √(budget·vw)`, then fill the budget along M. Snap each tile to
        // a divisor of its axis: a ragged tile masks every leaf and de-unrolls the kernel (~2×),
        // so a smaller clean tile wins. A narrow axis collapses its tile to the whole axis.
        let budget_elems = (ACC_REG_BYTES / elem).max(1);
        let tn_target = ((budget_elems * vw) as f64).sqrt() as usize;
        let tile_n = divisor_at_most(n, tn_target.max(1));
        let tile_m = divisor_at_most(m, (budget_elems / tile_n).max(1));

        // K depth: `alpha` lerps from a shallow `vw` to the deepest panel that keeps A
        // (tile_m×tile_k) and B (tile_k×tile_n) in L1 with the C tile (tile_m×tile_n)
        // resident, then snaps to a divisor of `k`. A ragged K tile bounds-checks every leaf
        // and disables the register unroll (~2×), so a clean shallower tile beats a deep
        // ragged one.
        let l1_tk = (L1_BYTES / elem).saturating_sub(tile_m * tile_n) / (tile_m + tile_n);
        let tk_cap = (vw + (alpha * l1_tk.saturating_sub(vw) as f32) as usize).clamp(1, k.max(1));
        let tile_k = divisor_at_most(k, tk_cap);

        // Plane grid: split the leaf grid among ~`cores` worker threads by aspect ratio.
        // Each plane is a thread and the cube loop is *serial*, so the factors must divide
        // the grid — an indivisible split inflates the cube count (serial depth) and idles
        // planes on the overhang. Snap the aspect-ratio target to grid divisors.
        let grid_m = m.div_ceil(tile_m).max(1);
        let grid_n = n.div_ceil(tile_n).max(1);
        let target_m = (cores as f64 * grid_m as f64 / grid_n as f64)
            .sqrt()
            .round() as usize;
        let plane_m = nearest_divisor(grid_m, target_m);
        let plane_n = nearest_divisor(grid_n, (cores / plane_m).max(1));

        // Tiles already divide their axes; the clamp is a defensive [1, axis] floor.
        let instruction = Instruction {
            m: tile_m.clamp(1, m.max(1)),
            n: tile_n.clamp(1, n.max(1)),
            k: tile_k.clamp(1, k.max(1)),
        };

        let planes = PlaneGrid {
            m: plane_m,
            n: plane_n,
        };

        CpuGemmBlueprint {
            instruction,
            planes,
        }
    }
}
