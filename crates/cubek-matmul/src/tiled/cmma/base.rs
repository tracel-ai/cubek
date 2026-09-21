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

/// Stages the ring keeps in flight: one filling while the one before it contracts.
const BUFFERING: usize = 2;

/// Units a cube runs, whatever the device would allow beyond it: past this a cube's planes
/// contend for one register file and the stage stops paying for the planes that fill it.
const MAX_UNITS_PER_CUBE: usize = 256;

/// Accumulator bytes one lane holds, where the device reports a register file this family's
/// size ([`reuse`]). Half of 256 32-bit registers a lane, the other half left to the operand
/// fragments, the addresses and the fill.
///
/// A budget and not a count, because a count means a different thing at each instruction: the
/// same 16 fragments are 512 bytes a lane at `16x16` accumulating in `f32` and 128 at `8x8`.
const ACCUMULATOR_LANE_BYTES: usize = 512;

/// Cubes a stage has to leave per streaming multiprocessor before it is worth its edges.
///
/// A cube runs to completion on one SM, so a grid that is not a whole number of waves pays for
/// a last one that is mostly idle, and the fewer waves there are the more of the run that is.
/// Measured on an L4 (58 SMs, f16): at 1024³ the widest stage leaves 1.1 waves and runs 0.177 ms
/// where a quarter of it leaves 4.4 and runs 0.152; at 1536³, 1.2 waves and 0.235 ms against 2.5
/// and 0.222. Past about three cubes an SM the tail stops showing and the wider stage wins on
/// bytes alone — 4096³ runs 6.09 ms at 17.7 waves against 7.32 at 70.6.
const CUBES_PER_SM_FLOOR: usize = 3;

/// Instruction tiles one cube's stage spans on one axis, at most. The stage is what the fill
/// moves per `K` step, and one too wide leaves fewer cubes than the device has places to run
/// them.
const MAX_TILES_PER_AXIS: usize = 32;

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

/// What the device has to keep busy: the cubes one launch of this problem runs, over the
/// places it has to run them. `None` where the runtime reports no SM count, which is every
/// backend but CUDA — there a stage is sized on bytes alone, as it was.
#[derive(Clone, Copy, Debug)]
struct Machine {
    sms: Option<usize>,
    boxes: usize,
}

impl Machine {
    /// Whether a `stages_m x stages_n` grid of cubes, one per box, keeps this machine busy.
    fn fills(&self, stages_m: usize, stages_n: usize) -> bool {
        match self.sms {
            Some(sms) => stages_m * stages_n * self.boxes >= sms * CUBES_PER_SM_FLOOR,
            None => true,
        }
    }
}

/// The instruction tiles a stage spans, grown onto whichever axis is currently the shorter so
/// the stage comes out as square as the two grids admit, and never past what `tiles` allows or
/// what the grid divides.
fn grow(tiles: usize, grid: (usize, usize)) -> (usize, usize) {
    let (grid_m, grid_n) = grid;
    let (mut tiles_m, mut tiles_n) = (1usize, 1usize);
    while tiles_m * tiles_n < tiles {
        let shorter_first = tiles_m <= tiles_n;
        let mut grew = false;
        for on_m in [shorter_first, !shorter_first] {
            let (own, other, edge) = match on_m {
                true => (tiles_m, tiles_n, grid_m),
                false => (tiles_n, tiles_m, grid_n),
            };
            let doubled = own * 2;
            if doubled > MAX_TILES_PER_AXIS
                || !edge.is_multiple_of(doubled)
                || doubled * other > tiles
            {
                continue;
            }
            match on_m {
                true => tiles_m = doubled,
                false => tiles_n = doubled,
            }
            grew = true;
            break;
        }
        if !grew {
            break;
        }
    }
    (tiles_m, tiles_n)
}

/// The plan for a device that pays for per-plane reuse: `(part_m, part_n, planes_m, planes_n)`.
///
/// Two counts decide it and the rest is arithmetic. **The fragments a plane holds** is
/// [`ACCUMULATOR_LANE_BYTES`] over what one fragment costs a lane, since the accumulator is
/// resident across the whole `K` walk and a plane that spills reads its own sums back from
/// memory. **The planes a cube runs** is `budget`, the units it is given over a plane's width.
///
/// Their product is the instruction tiles the stage spans, and it is grown one doubling at a
/// time onto whichever axis is currently the shorter, so the stage comes out as square as the
/// two grids admit — a square stage is the one that reads fewest bytes for the products it
/// makes. Only then is it cut into planes and fragments, the planes taking the coarser cut,
/// which is what leaves each plane a rectangle of fragments rather than a row.
fn reuse(
    instruction: (usize, usize),
    acc: ElemType,
    plane_dim: usize,
    budget: usize,
    grid: (usize, usize),
    machine: Machine,
) -> (usize, usize, usize, usize) {
    let ((im, inn), (grid_m, grid_n)) = (instruction, grid);
    let fragment_lane_bytes = (im * inn * acc.size()).div_ceil(plane_dim).max(1);
    let fragments = (ACCUMULATOR_LANE_BYTES / fragment_lane_bytes).max(1);
    let planes = budget.max(1);

    // Grow the stage by doubling the shorter axis, skipping a doubling the grid does not
    // divide: a stage that does not divide the problem is one this routine cannot mask.
    //
    // Then, where the machine's width is known, give the budget back a halving at a time while
    // the stage leaves too few cubes to fill it ([`CUBES_PER_SM_FLOOR`]): a stage is only worth
    // its bytes if there are enough of them to keep every SM busy to the end.
    //
    // Never below one fragment a plane, which is where there is no reuse left to give: a shape
    // small enough to want that has already given back everything the guard is for, and a stage
    // of one instruction is not a stage this routine can descend.
    let mut tiles = fragments * planes;
    let (tiles_m, tiles_n) = loop {
        let (tiles_m, tiles_n) = grow(tiles, (grid_m, grid_n));
        if tiles <= planes || machine.fills(grid_m / tiles_m, grid_n / tiles_n) {
            break (tiles_m, tiles_n);
        }
        tiles /= 2;
    };

    // Cut the tiles into planes and fragments, the planes taking the coarser cut: a plane
    // per rectangle of the stage, and the rectangle is what its fragments cover.
    let (mut planes_m, mut planes_n) = (1usize, 1usize);
    while planes_m * planes_n < planes {
        let coarser_first = tiles_m / planes_m >= tiles_n / planes_n;
        let mut grew = false;
        for on_m in [coarser_first, !coarser_first] {
            let (own, tiles_here) = match on_m {
                true => (planes_m, tiles_m),
                false => (planes_n, tiles_n),
            };
            let doubled = own * 2;
            if doubled > MAX_PLANES_PER_AXIS || !tiles_here.is_multiple_of(doubled) {
                continue;
            }
            match on_m {
                true => planes_m = doubled,
                false => planes_n = doubled,
            }
            grew = true;
            break;
        }
        if !grew {
            break;
        }
    }

    (tiles_m / planes_m, tiles_n / planes_n, planes_m, planes_n)
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

        // What a plane keeps resident and how many planes tile the cube's stage — the one
        // trade this routine makes, and a trade whose answer is the device's.
        //
        // A plane's accumulator lives in its lanes' registers, so how many fragments are
        // worth holding is a register budget: inside one, each lhs fragment is read `part_n`
        // times and each rhs fragment `part_m` times before it is dropped, and the stage
        // those counts build reads that many fewer bytes per product. Where the device
        // reports a register file this family's size, that is what [`reuse`] spends.
        //
        // Where it does not, the plan stays the one an Apple GPU measured — a single row of
        // fragments, planes spread along `m` — because there threadgroup residency beat
        // per-plane reuse (cross-point 5.2 vs 3.6 TFLOPS against a 2x8 partition on
        // square_4096 f16). Two arms and not one derivation, since nothing measured says the
        // two families sit on one curve.
        let (grid_m, grid_n) = (problem.m / im.max(1), problem.n / inn.max(1));
        let max_units =
            (client.properties().hardware.max_units_per_cube as usize).min(MAX_UNITS_PER_CUBE);
        let budget = (max_units / plane_dim).max(1);
        let reuses = client.properties().hardware.num_tensor_cores.is_some();

        let (part_m, part_n, planes_m, planes_n) = if reuses {
            reuse(
                (im, inn),
                acc,
                plane_dim,
                budget,
                (grid_m, grid_n),
                Machine {
                    sms: client
                        .properties()
                        .hardware
                        .num_streaming_multiprocessors
                        .map(|sms| sms as usize),
                    boxes: problem.out_batches.iter().product::<usize>().max(1),
                },
            )
        } else {
            let rows = (budget / inn.div_ceil(4).max(1)).max(1);
            (
                1,
                divisor_at_most(grid_n.max(1), rows.min(MAX_PLANES_PER_AXIS)),
                divisor_at_most(grid_m.max(1), rows.min(MAX_PLANES_PER_AXIS)),
                1,
            )
        };
        // A stored operand names the stage on its axes, whatever the plan would have picked:
        // the tiles are then the storage tile's, and only how they are cut into planes and
        // fragments is still this plan's to say.
        //
        // A reuse plan cuts them so the two multiply back exactly — a stage that lost a tile
        // to a rounded division is not the tile the weight was packed to. The other arm hands
        // every tile to a plane, as it did before there was a partition to give them to, and
        // the budget check below is what catches a weight too tall for one cube.
        let (part_n, planes_n) = match (fixed_n, reuses) {
            (Some(stage_n), true) => {
                let tiles = (stage_n / inn).max(1);
                let planes_n = divisor_at_most(tiles, planes_n.min(MAX_PLANES_PER_AXIS));
                (tiles / planes_n, planes_n)
            }
            (Some(stage_n), false) => (stage_n / inn, planes_n),
            (None, _) => (part_n, planes_n),
        };
        let (part_m, planes_m) = match (fixed_m, reuses) {
            (Some(stage_m), true) => {
                let tiles = (stage_m / im).max(1);
                let planes_m = divisor_at_most(tiles, planes_m.min(MAX_PLANES_PER_AXIS));
                (tiles / planes_m, planes_m)
            }
            (Some(stage_m), false) => (part_m, stage_m / im),
            (None, _) => (part_m, planes_m),
        };
        if planes_m * planes_n > budget {
            return Err(MatmulSetupError::InvalidConfig(Box::new(format!(
                "Cmma: lhs is stored in storage tiles {} rows tall, {planes_m} planes of the \
                 {im}-row instruction, more than the {budget} planes a cube holds",
                fixed_m.unwrap_or(0)
            ))));
        }

        // Stage depth, snapped down to the deepest `d·ik` dividing `k`.
        //
        // Where the stage was cut for reuse it is deep enough to spend the shared memory the
        // device has and no deeper: the `K` a buffered stage of these edges holds. That is one
        // number rather than a knee, because the edges are already the ones the register budget
        // asked for, and what is left to decide is how much of the fill's own memory to use.
        //
        // Where it was not, the knee is the byte budget the Apple cross-point measured: an
        // `f32` operand's stage is twice an `f16`'s at equal depth, and an `f32` accumulator
        // spends twice the registers. On square_4096 (f32 acc) f16 operands peaked at sk32
        // (4.71 vs 4.53 at 64) and f32 at sk16 (3.67 vs 3.26 at 32, 2.31 at 64): both ~64
        // stage-K bytes per row. The old f16-accumulate wanted twice that (sk64 at 4.87).
        let cap = if reuses {
            let (stage_m, stage_n) = (planes_m * part_m * im, planes_n * part_n * inn);
            let row_bytes = (stage_m * d.lhs.size() + stage_n * d.rhs.size()).max(1);
            let smem = client.properties().hardware.max_shared_memory_size;
            // And never deeper than leaves the ring a region per slot: a stage that swallows
            // the whole `K` walk is one region, which is a pipeline with nothing in flight.
            (smem / BUFFERING / row_bytes)
                .min(problem.k / BUFFERING)
                .max(ik)
        } else {
            let stage_k_bytes = if acc.size() >= 4 { 64 } else { 128 };
            (stage_k_bytes / d.lhs.size().max(1)).max(ik)
        };
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
            buffering: BUFFERING,
            delivery,
        })
    }
}

/// [`reuse`] is host arithmetic over the device's own numbers, so what it decides is testable
/// without one. The cases are the ones the L4 sweep turned on.
#[cfg(test)]
mod reuse_tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A machine wide enough that no stage is ever shrunk for it: the wave floor is its own
    /// test, and these cases are about the budgets.
    const ROOMY: Machine = Machine {
        sms: None,
        boxes: 1,
    };

    /// An L4: a `16x16` instruction accumulating in `f32` over 32-lane planes, eight planes a
    /// cube, a square grid wide enough to take any stage. The sweep's winner on every shape
    /// it ran — `4x4` fragments a plane over a `4x2` plane grid, a 256x128 stage — falls out
    /// of the two budgets and nothing else.
    #[test]
    fn a_large_register_file_buys_a_square_stage_of_rectangular_partitions() {
        assert_eq!(reuse((16, 16), F32, 32, 8, (256, 256), ROOMY), (4, 4, 4, 2));
    }

    /// The accumulator is budgeted in bytes a lane, not in fragments: the same budget holds
    /// four times as many `8x8` fragments as `16x16` ones, because one costs a quarter as
    /// much. A count stated here would mean a different register load at each instruction.
    #[test]
    fn the_budget_is_bytes_a_lane_so_a_smaller_instruction_holds_more_of_them() {
        let (pm, pn, gm, gn) = reuse((16, 16), F32, 32, 8, (256, 256), ROOMY);
        let (qm, qn, hm, hn) = reuse((8, 8), F32, 32, 8, (256, 256), ROOMY);
        assert_eq!(
            pm * pn * 4,
            qm * qn,
            "a quarter the bytes, four times the count"
        );
        assert_eq!(
            (gm * gn, hm * hn),
            (8, 8),
            "the plane count is the cube's either way"
        );
    }

    /// Every stage it states divides the grid it was given, on both axes: this routine cannot
    /// mask an overhang, so a stage that does not divide is a plan that cannot run.
    #[test]
    fn the_stage_divides_the_grid_it_was_cut_from() {
        for grid_m in [1usize, 2, 3, 5, 12, 64, 256] {
            for grid_n in [1usize, 2, 3, 5, 12, 64, 256] {
                let (pm, pn, gm, gn) = reuse((16, 16), F32, 32, 8, (grid_m, grid_n), ROOMY);
                assert!(grid_m.is_multiple_of(pm * gm), "{grid_m} / {pm}x{gm}");
                assert!(grid_n.is_multiple_of(pn * gn), "{grid_n} / {pn}x{gn}");
                assert!(gm * gn <= 8, "no more planes than the cube holds");
                assert!(gm <= MAX_PLANES_PER_AXIS && gn <= MAX_PLANES_PER_AXIS);
            }
        }
    }

    /// A grid that divides on one axis only puts its tiles there, rather than giving up the
    /// stage: a prime `n` is a shape, not a reason to read four times the bytes.
    #[test]
    fn a_grid_that_divides_on_one_axis_only_grows_along_it() {
        let (pm, pn, gm, gn) = reuse((16, 16), F32, 32, 8, (32, 1), ROOMY);
        assert_eq!((pn, gn), (1, 1), "nothing divides along n");
        assert!(pm * gm > 1, "so the tiles went to m: {pm}x{gm}");
    }
}

/// The wave floor, which is the one thing [`Machine`] decides: a stage is given back a halving
/// at a time while the cubes it leaves would not fill the device.
#[cfg(test)]
mod machine_tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// An L4's 58 SMs against the three square shapes measured on it. The widest stage is 16x8
    /// tiles (256x128); at 4096 it stands, and the two small shapes give back two halvings.
    ///
    /// The floor is three cubes an SM and not two because of what the two cost each other:
    /// at three, 1024³ lands on the stage it measured fastest at (0.152 ms, against 0.166 for
    /// one halving and 0.177 for none) and 1536³ lands one halving past its own (0.230 against
    /// 0.222); at two, 1536³ is exact and 1024³ is 9% off instead of 3.6%. A floor is one
    /// number for every shape, so it is the worse of the two misses that picks it.
    #[test]
    fn a_narrow_grid_gives_the_stage_back_until_it_fills_the_device() {
        let l4 = |boxes| Machine {
            sms: Some(58),
            boxes,
        };
        let tiles = |grid: usize, boxes| {
            let (pm, pn, gm, gn) = reuse((16, 16), F32, 32, 8, (grid, grid), l4(boxes));
            (pm * gm * 16, pn * gn * 16)
        };
        assert_eq!(
            tiles(256, 2),
            (256, 128),
            "4096³: 17.7 waves, nothing given back"
        );
        assert_eq!(tiles(96, 1), (128, 64), "1536³");
        assert_eq!(tiles(64, 2), (128, 64), "1024³");
    }

    /// A device that reports no SM count is one this has nothing to say about, so the stage is
    /// sized on its bytes alone — which is every backend but CUDA today.
    #[test]
    fn a_machine_of_unknown_width_never_shrinks_a_stage() {
        let unknown = Machine {
            sms: None,
            boxes: 1,
        };
        let (pm, pn, gm, gn) = reuse((16, 16), F32, 32, 8, (64, 64), unknown);
        assert_eq!((pm * gm * 16, pn * gn * 16), (256, 128));
    }
}
