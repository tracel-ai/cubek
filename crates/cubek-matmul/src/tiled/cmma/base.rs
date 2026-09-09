//! The Cmma routine: the classic simple matmul (plane-partitioned stage, cooperative
//! cyclic loading, tensor-core leaf) ported onto the tile DSL.
//!
//! Each cube owns a `planes·partition·instruction`-sized stage along `m`/`n`; the walk
//! refills one smem stage per `K` step (depth smem-budgeted), filled cooperatively
//! (cyclic across the cube's units). Within the stage each plane owns a [`Partition`] of
//! instruction-sized cmma fragments, resident across the whole `K` walk.
//!
//! # Rejected (returns [`MatmulSetupError`])
//!
//! - Backends without a matching cmma [`MmaConfig`] (and `plane_size 1`, i.e. CPU).
//! - Quantized inputs.
//! - Operands not row-major contiguous (col-major needs a fragment-layout path not yet wired).
//! - Shapes not divisible by the instruction (the cmma transport cannot mask an overhang).
//! - A storage-tiled input whose storage tile is not this plan's stage on its axes: the storage
//!   tile names the stage, the plan cannot disagree. Which delivery moves it is unrelated.

use std::fmt::Display;

use cubecl::features::MmaConfig;
use cubecl::{features::Tma as TmaFeature, ir::ElemType};

use crate::{
    definition::{MatmulAvailabilityError, MatmulProblem, MatmulSetupError},
    routine::{BlueprintStrategy, DeviceSettings, Routine},
    tiled::cpu_gemm::{InstructionShape, PlaneGrid},
};

/// Upper bound on planes along one stage axis; 2×4 or 4×2 tends to saturate without
/// blowing the cube dim.
const MAX_PLANES_PER_AXIS: usize = 4;

/// The CMMA routine's launch-time input transport choice. This is deliberately separate from
/// [`cubek_tile::Delivery`], which describes an already-constructed tile's staging behavior.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum CmmaDelivery {
    /// The cube's units move each stage.
    #[default]
    Copy,
    /// The TMA engine moves each stage.
    Tma,
}

impl CmmaDelivery {
    pub(crate) fn is_tma(self) -> bool {
        matches!(self, CmmaDelivery::Tma)
    }

    fn validate_tma(self, boxes: &[usize], batched: bool) -> Result<(), String> {
        if self.is_tma() {
            cubek_tile::Delivery::Tma.validate_tma(boxes, batched)
        } else {
            Ok(())
        }
    }
}

/// Tiles per plane along `m`/`n`: the plane's resident fragment partition,
/// sized so `A`/`B` fragments are reused across executes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Partition {
    pub m: usize,
    pub n: usize,
}

/// A fully-resolved plan: the tensor-core [`InstructionShape`], each plane's fragment
/// [`Partition`], how many planes tile the cube's stage along `m`/`n` ([`PlaneGrid`]), how
/// deep each smem stage runs along `K` (`stage_k`) and how many stages are in flight
/// (`buffering`). The kernel's comptime argument, whose level methods both the launch and the
/// kernel's loops read.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CmmaBlueprint {
    pub instruction: InstructionShape,
    pub partition: Partition,
    pub planes: PlaneGrid,
    /// K-stage depth in elements: a multiple of `instruction.k`, chosen by [`select`]
    /// against the shared-memory budget.
    pub stage_k: usize,
    /// Stages in flight: `2` overlaps a stage's fill with the previous one's contraction.
    pub buffering: usize,
    /// Launch-time transport for both inputs (the output always uses a regular buffer copy).
    pub delivery: CmmaDelivery,
}

impl CmmaBlueprint {
    /// The cube's stage edges along `m`/`n`: with [`stage_k`](Self::stage_k), the storage tile a
    /// storage-tiled input is packed to.
    pub fn stage(&self) -> (usize, usize) {
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
            || self.buffering == 0
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

/// The storage tiles the operands arrived in, read off their bindings
/// ([`storage_tile`](crate::tiled::storage_tile)): each names the stage on its axes, since a
/// routine reads whole storage tiles. A plain operand names nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct StoredTiles {
    /// The lhs tile `(rows, cols)`, which is `(stage_m, stage_k)`.
    pub lhs: Option<(usize, usize)>,
    /// The rhs tile `(rows, cols)`, which is `(stage_k, stage_n)`.
    pub rhs: Option<(usize, usize)>,
}

impl StoredTiles {
    /// The stage edges these tiles fix, `(stage_m, stage_k, stage_n)`, `None` where no operand
    /// names one.
    ///
    /// # Errors
    ///
    /// Both operands stored, in different `K` runs: each names `stage_k`, so they must agree.
    #[allow(clippy::result_large_err, clippy::type_complexity)]
    fn stage(self) -> Result<(Option<usize>, Option<usize>, Option<usize>), MatmulSetupError> {
        let stage_k = match (self.lhs, self.rhs) {
            (Some((_, lk)), Some((rk, _))) if lk != rk => {
                return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                    "Cmma: lhs is stored in storage tiles {lk} deep along K and rhs {rk} deep; \
                     both name stage_k, so pack them to the same depth"
                ))));
            }
            (Some((_, k)), _) | (None, Some((k, _))) => Some(k),
            (None, None) => None,
        };
        Ok((self.lhs.map(|(m, _)| m), stage_k, self.rhs.map(|(_, n)| n)))
    }
}

/// The routine's launch knobs; the geometry is fully inferred.
#[derive(Clone, Debug, Default)]
pub struct CmmaStrategy {
    /// How the operands' bytes move (the output is always strided). `Tma` is
    /// `Unavailable` on backends without the feature; it never silently degrades.
    pub delivery: CmmaDelivery,
}

impl CmmaStrategy {
    pub fn tma() -> Self {
        CmmaStrategy {
            delivery: CmmaDelivery::Tma,
        }
    }
}

impl Display for CmmaStrategy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self.delivery {
            CmmaDelivery::Copy => Ok(()),
            CmmaDelivery::Tma => f.write_str("_tma"),
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
    /// Resolve `strategy` into a validated plan for `problem` on this device. `acc` is the
    /// register accumulate type (e.g. `f32` under an `f16` output); the selected
    /// instruction's [`MmaConfig`] is keyed on it, since that is the accumulator the
    /// kernel emits.
    #[allow(clippy::result_large_err)]
    ///
    /// `stored` is what the operands' storage tiles fix: an inferred plan stages to them; a
    /// forced plan is checked against them at the launch.
    pub fn blueprint(
        strategy: &BlueprintStrategy<(), CmmaRoutine>,
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        acc: ElemType,
        stored: StoredTiles,
    ) -> Result<CmmaBlueprint, MatmulSetupError> {
        let blueprint = match strategy {
            BlueprintStrategy::Forced(blueprint) => blueprint.clone(),
            BlueprintStrategy::Inferred(args) => {
                Self::select(problem, device_settings, args.delivery, acc, stored)?
            }
        };
        // Pure plan validation first (backend-independent), the availability gate after.
        // The gate covers forced plans too: they skip `select`'s hardware stages, and an
        // unsupported instruction must be `Unavailable`, not garbage.
        blueprint.validate(problem)?;
        let plane_dim = device_settings.client.properties().hardware.plane_size_max;
        if plane_dim <= 1 {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnsupported { plane_dim },
            ));
        }
        let (i, d) = (blueprint.instruction, &problem.global_dtypes);
        if !device_settings
            .client
            .properties()
            .features
            .matmul
            .cmma
            .contains(&MmaConfig {
                a_type: d.lhs,
                b_type: d.rhs,
                cd_type: acc,
                m: i.m as u32,
                n: i.n as u32,
                k: i.k as u32,
            })
        {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TileSizeNotFound,
            ));
        }
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
    /// classic `find_instruction_size` stages), then tile the stage with as many planes as
    /// the cube dim affords, snapped to divisors of the tile grid. A stage edge a stored
    /// operand fixes is taken as is: the instruction must divide it and the planes and
    /// partition are sized to realize it exactly, whatever this problem's `m` would have chosen,
    /// since a weight is packed once and read at every `m`.
    #[allow(clippy::result_large_err)]
    fn select(
        problem: &MatmulProblem,
        device_settings: &DeviceSettings,
        delivery: CmmaDelivery,
        acc: ElemType,
        stored: StoredTiles,
    ) -> Result<CmmaBlueprint, MatmulSetupError> {
        let client = &device_settings.client;
        // The storage tiles constrain the plan's stage; they do not touch the delivery, which
        // says who moves the bytes and serves a stored operand either way.
        let (fixed_m, fixed_k, fixed_n) = stored.stage()?;
        let divides = |edge: Option<usize>, i: usize| edge.is_none_or(|e| e.is_multiple_of(i));
        let fits = |m: usize, n: usize, k: usize| {
            divides(fixed_m, m) && divides(fixed_n, n) && divides(fixed_k, k)
        };
        let plane_dim = client.properties().hardware.plane_size_max as usize;
        if plane_dim <= 1 {
            return Err(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::PlaneDimUnsupported {
                    plane_dim: plane_dim as u32,
                },
            ));
        }

        // The config is keyed on `acc` (the register accumulator the kernel emits), not
        // the stored output `d.out`: for an f16 output the hardware config is
        // a=f16,b=f16,cd=f32, and the epilogue casts f32 down on drain.
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
                    cd_type: acc,
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
            .find(|&(m, n, k)| supported(m, n, k) && fits(m, n, k))
            .or_else(|| {
                client
                    .properties()
                    .features
                    .matmul
                    .cmma
                    .iter()
                    .find(|c| c.a_type == d.lhs && c.b_type == d.rhs && c.cd_type == acc)
                    .map(|c| (c.m as usize, c.n as usize, c.k as usize))
                    .filter(|&(m, n, k)| fits(m, n, k))
            })
            .ok_or(MatmulSetupError::Unavailable(
                MatmulAvailabilityError::TileSizeNotFound,
            ))?;

        // The thin shape (single-row partitions, planes along `m`, small stages): high
        // threadgroup residency beats per-plane reuse on Metal. Cross-point measured
        // 5.2 vs 3.6 TFLOPS against the old fat 2x8 selection on square_4096 f16.
        let (grid_m, grid_n) = (problem.m / im.max(1), problem.n / inn.max(1));
        let max_units = (client.properties().hardware.max_units_per_cube as usize).min(256);
        let budget = (max_units / plane_dim).max(1);
        let rows = (budget / inn.div_ceil(4).max(1)).max(1);

        let part_m = 1;
        let part_n = match fixed_n {
            Some(stage_n) => stage_n / inn,
            None => divisor_at_most(grid_n.max(1), rows.min(MAX_PLANES_PER_AXIS)),
        };
        let planes_m = match fixed_m {
            Some(stage_m) => stage_m / im,
            None => divisor_at_most(grid_m.max(1), rows.min(MAX_PLANES_PER_AXIS)),
        };
        let planes_n = 1;
        if planes_m * planes_n > budget {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Cmma: lhs is stored in storage tiles {} rows tall, {planes_m} planes of the \
                 {im}-row instruction, more than the {budget} planes a cube holds",
                fixed_m.unwrap_or(0)
            ))));
        }

        // Stage depth, snapped down to the deepest `d·ik` dividing `k`. The knee is set by
        // the double-buffered smem the cooperative fill must keep resident, so it scales by
        // a *byte* budget: an `f32` operand's stage is twice an `f16`'s at equal depth. An
        // `f32` accumulator (always now, since tensor cores accumulate in `f32`) also spends
        // twice the registers, tightening the budget vs the old `f16` accumulate. Measured
        // on square_4096 (f32 acc): f16 operands peak at sk32 (4.71 vs 4.53 at 64), f32 at
        // sk16 (3.67 vs 3.26 at 32, 2.31 at 64): both ~64 stage-K bytes per row. The old
        // f16-accumulate wanted twice that (sk64 at 4.87).
        let stage_k_bytes = if acc.size() >= 4 { 64 } else { 128 };
        let cap = (stage_k_bytes / d.lhs.size().max(1)).max(ik);
        let stage_k = match fixed_k {
            Some(stage_k) => stage_k,
            None => (1..=(cap / ik).max(1))
                .rev()
                .map(|d| d * ik)
                .find(|&sk| problem.k.is_multiple_of(sk))
                .unwrap_or(ik),
        };

        Ok(CmmaBlueprint {
            instruction: InstructionShape {
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
            buffering: 2,
            delivery,
        })
    }
}
