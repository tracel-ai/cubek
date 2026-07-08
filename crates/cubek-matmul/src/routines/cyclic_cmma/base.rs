//! The CyclicCmma routine: the classic simple matmul (plane-partitioned stage, cooperative
//! cyclic loading, tensor-core leaf) ported onto the tile DSL.
//!
//! Each cube owns a `planes.m·instruction.m × planes.n·instruction.n` stage; a
//! double-buffered walk rotates `instruction.k`-deep smem slots along `K`, filled
//! cooperatively (cyclic across the cube's units). Within the stage each plane owns one
//! `instruction`-sized cmma fragment, resident across the whole `K` walk.
//!
//! # Rejected (returns [`MatmulSetupError`])
//!
//! - Backends without a matching cmma [`MmaConfig`] (and `plane_size 1`, i.e. CPU).
//! - Quantized inputs.
//! - Operands not row-major contiguous: the cmma transport addresses a window by row
//!   stride, so col-major needs a fragment-layout path not yet wired.
//! - Shapes not divisible by the instruction (`m % i.m`, `n % i.n`, `k % i.k`): the cmma
//!   transport cannot mask an overhang.

use std::fmt::Display;

use cubecl::{Runtime, features::MmaConfig};
use cubek_tile::Axis;

use crate::{
    definition::{MatmulAvailabilityError, MatmulProblem, MatmulSetupError},
    routines::{
        BlueprintStrategy, DeviceSettings, Routine,
        cpu_gemm::{Instruction, PlaneGrid},
    },
};

// Matmul axes
pub(crate) const M: Axis = Axis(0);
pub(crate) const N: Axis = Axis(1);
pub(crate) const K: Axis = Axis(2);
/// The axis for output batch dimension `i` (outermost is `0`).
pub(crate) fn batch_axis(i: usize) -> Axis {
    Axis(3 + i as u8)
}

/// Upper bound on planes along one stage axis; 2×4 or 4×2 tends to saturate without
/// blowing the cube dim.
const MAX_PLANES_PER_AXIS: usize = 4;

/// A fully-resolved plan: the tensor-core [`Instruction`] each plane executes and how many
/// planes tile the cube's stage along `m`/`n` ([`PlaneGrid`]).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicCmmaBlueprint {
    pub instruction: Instruction,
    pub planes: PlaneGrid,
}

impl CyclicCmmaBlueprint {
    /// Reject a plan this routine cannot run: a degenerate cuboid, or a shape the cmma
    /// transport would have to mask.
    #[allow(clippy::result_large_err)]
    pub fn validate(&self, problem: &MatmulProblem) -> Result<(), MatmulSetupError> {
        let (i, p) = (self.instruction, self.planes);
        if i.m == 0 || i.n == 0 || i.k == 0 || p.m == 0 || p.n == 0 {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CyclicCmma blueprint must be non-zero, got instruction {}x{}x{} planes {}x{}",
                i.m, i.n, i.k, p.m, p.n
            ))));
        }
        let (stage_m, stage_n) = (p.m * i.m, p.n * i.n);
        if !problem.m.is_multiple_of(stage_m)
            || !problem.n.is_multiple_of(stage_n)
            || !problem.k.is_multiple_of(i.k)
        {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "CyclicCmma requires a shape divisible by the stage: \
                 {}x{}x{} vs stage {stage_m}x{stage_n}x{}",
                problem.m, problem.n, problem.k, i.k
            ))));
        }
        Ok(())
    }
}

/// No knobs yet; the selection is fully inferred.
#[derive(Clone, Debug, Default)]
pub struct CyclicCmmaStrategy;

impl Display for CyclicCmmaStrategy {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        Ok(())
    }
}

/// Pairs the [`CyclicCmmaStrategy`] knob with the [`CyclicCmmaBlueprint`] plan.
pub struct CyclicCmmaRoutine;

impl Routine<()> for CyclicCmmaRoutine {
    type Strategy = CyclicCmmaStrategy;
    type Blueprint = CyclicCmmaBlueprint;
}

/// The largest divisor of `g` not exceeding `cap` (≥1).
fn divisor_at_most(g: usize, cap: usize) -> usize {
    let cap = cap.clamp(1, g.max(1));
    (1..=cap).rev().find(|d| g.is_multiple_of(*d)).unwrap_or(1)
}

impl CyclicCmmaRoutine {
    /// Resolve `strategy` into a validated plan for `problem` on this device.
    #[allow(clippy::result_large_err)]
    pub fn blueprint<R: Runtime>(
        strategy: &BlueprintStrategy<(), CyclicCmmaRoutine>,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> Result<CyclicCmmaBlueprint, MatmulSetupError> {
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(_) => Self::select(problem, device_settings)?,
        };
        blueprint.validate(problem)?;
        Ok(blueprint)
    }

    /// Pick the instruction from the hardware's cmma configs (aspect-aware, mirroring the
    /// classic `find_instruction_size` ladder), then tile the stage with as many planes as
    /// the cube dim affords, snapped to divisors of the tile grid.
    #[allow(clippy::result_large_err)]
    fn select<R: Runtime>(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings<R>,
    ) -> Result<CyclicCmmaBlueprint, MatmulSetupError> {
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
            client.properties().features.matmul.cmma.contains(&MmaConfig {
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

        // Plane grid: fill the units-per-cube budget along m then n, snapped to divisors
        // of the tile grid so the stage never overhangs (cmma cannot mask).
        let max_units = (client.properties().hardware.max_units_per_cube as usize).min(256);
        let budget = (max_units / plane_dim).max(1);
        let (grid_m, grid_n) = (problem.m / im.max(1), problem.n / inn.max(1));
        let planes_m = divisor_at_most(grid_m.max(1), budget.min(MAX_PLANES_PER_AXIS));
        let planes_n = divisor_at_most(grid_n.max(1), (budget / planes_m).min(MAX_PLANES_PER_AXIS));

        Ok(CyclicCmmaBlueprint {
            instruction: Instruction {
                m: im,
                n: inn,
                k: ik,
            },
            planes: PlaneGrid {
                m: planes_m,
                n: planes_n,
            },
        })
    }
}
