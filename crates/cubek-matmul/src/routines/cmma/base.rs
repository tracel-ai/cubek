//! The Cmma routine: the classic simple matmul (plane-partitioned stage, cooperative
//! cyclic loading, tensor-core leaf) ported onto the tile DSL.
//!
//! Each cube owns a `planes·partition·instruction`-sized stage along `m`/`n`; a
//! double-buffered walk rotates smem slots along `K` (depth smem-budgeted), filled
//! cooperatively (cyclic across the cube's units). Within the stage each plane owns a
//! [`Partition`] of instruction-sized cmma fragments, resident across the whole `K` walk.
//!
//! # Rejected (returns [`MatmulSetupError`])
//!
//! - Backends without a matching cmma [`MmaConfig`] (and `plane_size 1`, i.e. CPU).
//! - Quantized inputs.
//! - Operands not row-major contiguous (col-major needs a fragment-layout path not yet wired).
//! - Shapes not divisible by the instruction (the cmma transport cannot mask an overhang).

use std::fmt::Display;

use cubecl::features::Tma as TmaFeature;
use cubecl::{Runtime, features::MmaConfig};
use cubek_tile::Delivery;

use crate::{
    definition::{MatmulAvailabilityError, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy, DeviceSettings, Routine,
        cpu_gemm::{Instruction, PlaneGrid},
    },
};

/// Upper bound on planes along one stage axis; 2×4 or 4×2 tends to saturate without
/// blowing the cube dim.
const MAX_PLANES_PER_AXIS: usize = 4;

/// Tiles per plane along `m`/`n`: the plane's resident fragment partition,
/// sized so `A`/`B` fragments are reused across executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Partition {
    pub m: usize,
    pub n: usize,
}

/// A fully-resolved plan: the tensor-core [`Instruction`], each plane's fragment
/// [`Partition`], how many planes tile the cube's stage along `m`/`n` ([`PlaneGrid`]), and how
/// deep each double-buffered smem stage runs along `K` (`stage_k`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CmmaBlueprint {
    pub instruction: Instruction,
    pub partition: Partition,
    pub planes: PlaneGrid,
    /// K-stage depth in elements: a multiple of `instruction.k`, chosen by [`select`]
    /// against the shared-memory budget.
    pub stage_k: usize,
    /// How the operands' bytes move (the output is always strided).
    pub delivery: Delivery,
}

impl CmmaBlueprint {
    /// The cube's stage edges along `m`/`n`.
    pub(crate) fn stage(&self) -> (usize, usize) {
        (
            self.planes.m * self.partition.m * self.instruction.m,
            self.planes.n * self.partition.n * self.instruction.n,
        )
    }

    /// Reject a plan this routine cannot run: a degenerate cuboid, or a shape the cmma
    /// transport would have to mask.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        let (i, c, p) = (self.instruction, self.partition, self.planes);
        if i.m == 0
            || i.n == 0
            || i.k == 0
            || c.m == 0
            || c.n == 0
            || p.m == 0
            || p.n == 0
            || self.stage_k == 0
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Cmma blueprint must be non-zero, got instruction {}x{}x{} \
                 partition {}x{} planes {}x{} stage_k {}",
                i.m, i.n, i.k, c.m, c.n, p.m, p.n, self.stage_k
            ))));
        }
        let (stage_m, stage_n) = self.stage();
        if !problem.m.is_multiple_of(stage_m)
            || !problem.n.is_multiple_of(stage_n)
            || !problem.k.is_multiple_of(self.stage_k)
            || !self.stage_k.is_multiple_of(i.k)
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Cmma requires a shape divisible by the stage: \
                 {}x{}x{} vs stage {stage_m}x{stage_n}x{} (stage_k {})",
                problem.m, problem.n, problem.k, i.k, self.stage_k
            ))));
        }
        // The bulk-copy box is the stage; TMA owns which boxes it can encode.
        let batched = problem.out_batches.iter().any(|&b| b > 1);
        self.delivery
            .validate_tma(&[stage_m, stage_n, self.stage_k], batched)
            .map_err(|e| MatmulSetupError::InvalidConfig(Box::new(e)))?;
        Ok(())
    }
}

/// The routine's launch knobs; the geometry is fully inferred.
#[derive(Clone, Debug, Default)]
pub struct CmmaStrategy {
    /// How the operands' bytes move (the output is always strided). `Tma` is
    /// `Unavailable` on backends without the feature; it never silently degrades.
    pub delivery: Delivery,
}

impl CmmaStrategy {
    pub fn tma() -> Self {
        CmmaStrategy {
            delivery: Delivery::Tma,
        }
    }
}

impl Display for CmmaStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.delivery {
            Delivery::Strided => Ok(()),
            Delivery::Tma => f.write_str("_tma"),
        }
    }
}

/// Pairs the [`CmmaStrategy`] knob with the [`CmmaBlueprint`] plan.
pub struct CmmaRoutine;

impl Routine<()> for CmmaRoutine {
    type Strategy = CmmaStrategy;
    type Blueprint = CmmaBlueprint;
}

/// The largest divisor of `g` not exceeding `cap` (≥1).
fn divisor_at_most(g: usize, cap: usize) -> usize {
    let cap = cap.clamp(1, g.max(1));
    (1..=cap).rev().find(|d| g.is_multiple_of(*d)).unwrap_or(1)
}

impl CmmaRoutine {
    /// Resolve `strategy` into a validated plan for `problem` on this device.
    #[allow(clippy::result_large_err)]
    pub fn blueprint<R: Runtime>(
        strategy: &BlueprintStrategy<(), CmmaRoutine>,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> Result<CmmaBlueprint, MatmulSetupError> {
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(args) => {
                Self::select(problem, device_settings, args.delivery)?
            }
        };
        // Pure plan validation first (backend-independent), the availability gate after.
        blueprint.validate(problem)?;
        if blueprint.delivery.is_tma()
            && !device_settings
                .client
                .properties()
                .features
                .tma
                .contains(TmaFeature::Base)
        {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TmaUnavailable,
            ));
        }
        Ok(blueprint)
    }

    /// Pick the instruction from the hardware's cmma configs (aspect-aware, mirroring the
    /// classic `find_instruction_size` ladder), then tile the stage with as many planes as
    /// the cube dim affords, snapped to divisors of the tile grid.
    #[allow(clippy::result_large_err)]
    fn select<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
        delivery: Delivery,
    ) -> Result<CmmaBlueprint, MatmulSetupError> {
        let client = &device_settings.client;
        let plane_dim = client.properties().hardware.plane_size_max as usize;
        if plane_dim <= 1 {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnsupported {
                    plane_dim: plane_dim as u32,
                },
            ));
        }

        let d = &problem.global_dtypes;
        let supported = |m: usize, n: usize, k: usize| {
            client
                .properties()
                .features
                .matmul
                .cmma
                .contains(&MmaConfig {
                    a_type: d.lhs,
                    b_type: d.rhs,
                    cd_type: d.out,
                    m: m as u32,
                    n: n as u32,
                    k: k as u32,
                })
        };

        let (m, n) = (problem.m, problem.n);
        let candidates: &[(usize, usize, usize)] = if m >= 4 * n {
            &[(32, 8, 16), (16, 16, 16), (8, 8, 8)]
        } else if n >= 4 * m {
            &[(8, 32, 16), (16, 16, 16), (8, 8, 8)]
        } else {
            &[(16, 16, 16), (8, 8, 8)]
        };
        let (im, inn, ik) = candidates
            .iter()
            .copied()
            .find(|&(m, n, k)| supported(m, n, k))
            .or_else(|| {
                client
                    .properties()
                    .features
                    .matmul
                    .cmma
                    .iter()
                    .find(|c| c.a_type == d.lhs && c.b_type == d.rhs && c.cd_type == d.out)
                    .map(|c| (c.m as usize, c.n as usize, c.k as usize))
            })
            .ok_or(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TileSizeNotFound,
            ))?;

        // Each plane's partition, snapped to divisors of the tile grid so the stage
        // never overhangs. The legacy shape: partition_n ≈ 8, rows 2.
        let (grid_m, grid_n) = (problem.m / im.max(1), problem.n / inn.max(1));
        let part_n = divisor_at_most(grid_n.max(1), 8);
        let part_m = divisor_at_most(grid_m.max(1), 2);

        // Plane grid over the remaining partition grid, capped at 256 units like the
        // legacy selector: past it Metal rejects the cube dim and silently zeroes output.
        let max_units = (client.properties().hardware.max_units_per_cube as usize).min(256);
        let budget = (max_units / plane_dim).max(1);
        let planes_m = divisor_at_most((grid_m / part_m).max(1), budget.min(MAX_PLANES_PER_AXIS));
        let planes_n = divisor_at_most(
            (grid_n / part_n).max(1),
            (budget / planes_m).min(MAX_PLANES_PER_AXIS),
        );

        // Stage depth: the deepest `d·ik` dividing `k` (d ≤ 8) whose two
        // double-buffered slots still fit shared memory.
        let (stage_m, stage_n) = (planes_m * part_m * im, planes_n * part_n * inn);
        let smem_budget = 32 * 1024;
        let row_bytes = stage_m * d.lhs.size() + stage_n * d.rhs.size();
        let stage_k = (1..=8usize)
            .rev()
            .map(|d| d * ik)
            .find(|&sk| problem.k.is_multiple_of(sk) && 2 * sk * row_bytes <= smem_budget)
            .unwrap_or(ik);

        Ok(CmmaBlueprint {
            instruction: Instruction {
                m: im,
                n: inn,
                k: ik,
            },
            partition: Partition {
                m: part_m,
                n: part_n,
            },
            planes: PlaneGrid {
                m: planes_m,
                n: planes_n,
            },
            stage_k,
            delivery,
        })
    }
}
