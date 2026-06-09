use std::fmt::Display;

use cubecl::Runtime;

use crate::{
    definition::{MatmulProblem, MatmulSetupError},
    routines::{BlueprintStrategy, DeviceSettings, Routine},
};

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
    /// Reject a blueprint whose blocks don't tile the problem evenly. Temporary: edge
    /// tiles aren't masked yet, so each block must divide its axis.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        let (m, n, k) = (problem.m, problem.n, problem.k);
        if !m.is_multiple_of(self.tile_m)
            || !n.is_multiple_of(self.tile_n)
            || !k.is_multiple_of(self.tile_k)
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CpuGemm blocks {}x{}x{} must divide M={m}, N={n}, K={k}",
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

        // N rides SIMD lines
        let mut tile_n = edge.div_ceil(vw) * vw.clamp(vw, n.max(1));
        // M is register rows
        let mut tile_m = edge.clamp(1, m.max(1));

        // Parallelism floor: keep at least one cube per core. Utilisation overrides the
        // `alpha` preference for reuse.
        while batch * m.div_ceil(tile_m) * n.div_ceil(tile_n) < cores && (tile_m > 1 || tile_n > vw)
        {
            if tile_m >= tile_n {
                tile_m = (tile_m / 2).max(1);
            } else {
                tile_n = (tile_n / 2).max(vw);
            }
        }

        // K depth: fill the rest of L1 with the A (tile_m×tile_k) and B (tile_k×tile_n)
        // panels while the C tile (tile_m×tile_n) stays resident.
        let tile_k = ((L1_BYTES / elem).saturating_sub(tile_m * tile_n) / (tile_m + tile_n))
            .clamp(1, k.max(1));

        // TODO(masking): edge tiles aren't masked yet, so each block must divide its axis.
        // Drop these snaps once partial-tile masking lands — the choice above already
        // targets the mask-free ideal.
        CpuGemmBlueprint {
            tile_m: largest_divisor_at_most(m, tile_m),
            tile_n: largest_divisor_at_most(n, tile_n),
            tile_k: largest_divisor_at_most(k, tile_k),
        }
    }
}

/// Largest divisor of `n` that is `≤ cap` (at least 1).
fn largest_divisor_at_most(n: usize, cap: usize) -> usize {
    let cap = cap.clamp(1, n.max(1));
    (1..=cap).rev().find(|d| n % d == 0).unwrap_or(1)
}
