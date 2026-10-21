use super::kernel::plane_shuffle_matmul_kernel;
use crate::kernels::MatmulBinding;
use crate::kernels::{
    MATMUL_Q_RHS_NOT_PACKED, MatmulBlueprint, MatmulOutput, MatmulRoutine, MatmulRoutineProblem,
    MatmulTensors, MatmulView,
};
use crate::microkernel::MatmulPartitioning;
use crate::microkernel::dense::plane::{PlaneShuffleMicrokernel, PlaneShuffleMicrokernelSettings};
use crate::microkernel::{MatmulOperand, MatmulOrder};
use crate::stored::Packing;
use core::num::NonZeroUsize;
use cubek_std::Hardware;
use cubek_std::matmul_axes::{K, M, N, blocked};
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, ROUTINE_LISTS_NO_SETTING, Routine};
use cubek_std::selector::divisor_at_most;
use cubek_tile::launch::DestinationLaunch;
use cubek_tile::launch::Refusal;
use cubek_tile::launch::Vectorization;
use cubek_tile::space::{CubeOrder, PlaneOrder};
use cubek_tile::{Axis, TileArgLaunch};

/// `out = lhs · rhs` with every plane of the launch owning every row of a run of the output's
/// columns, the launch grouping its planes in cubes side by side along `n`. Each column is read
/// once, by the one plane that owns it.
///
/// It runs where the device commits to a plane width, since its units add their sums across the plane.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneShuffleMatmulRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleMatmulRoutineSettings {
    pub microkernel: PlaneShuffleMicrokernelSettings,
    /// Planes one cube holds, side by side along `n`.
    pub planes: NonZeroUsize,
}

impl PlaneShuffleMatmulRoutineSettings {
    /// A step of 16 read by the whole plane, four planes a cube: between the small steps and few
    /// planes measured best at decode on NVIDIA and the long steps and many planes measured best
    /// on ROCm.
    pub const DEFAULT: Self = Self {
        microkernel: PlaneShuffleMicrokernelSettings::DEFAULT,
        planes: NonZeroUsize::new(4).unwrap(),
    };

    /// One step a unit, four planes a cube: the lead against a packed weight, whose decode the
    /// weight dequantized first (its caller's dequantize routine) places its planes at.
    pub const PACKED: Self = Self {
        microkernel: PlaneShuffleMicrokernelSettings::stepping(1, None),
        planes: NonZeroUsize::new(4).unwrap(),
    };
}

impl core::fmt::Display for PlaneShuffleMatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "plane-shuffle-matmul {} planes={}",
            self.microkernel, self.planes
        )
    }
}

pub type PlaneShuffleMatmulRoutineBlueprint = MatmulBlueprint<PlaneShuffleMicrokernel>;

impl PlaneShuffleMatmulRoutine {
    /// The settings worth racing against a packed weight: the lead
    /// ([`PACKED`](PlaneShuffleMatmulRoutineSettings::PACKED)), then the settings measured best, or
    /// within five percent of it, on the Qwen3-4B projections at one, four and eight rows on an
    /// RTX PRO 6000, the weight in rows and in tile order: each a count of steps and of rows a unit
    /// ([`PlaneShuffleMicrokernelSettings::stepping`]) on a count of planes a cube. What the
    /// packed few-rows table races, and what a caller naming the routine and no setting runs
    /// against a packed weight, in this order.
    pub fn packed_settings(&self) -> Vec<PlaneShuffleMatmulRoutineSettings> {
        use PlaneShuffleMicrokernelSettings as Block;
        let on = |microkernel, planes| PlaneShuffleMatmulRoutineSettings {
            microkernel,
            planes: NonZeroUsize::new(planes).unwrap(),
        };
        vec![
            PlaneShuffleMatmulRoutineSettings::PACKED,
            on(Block::stepping(1, None), 1),
            on(Block::stepping(2, None), 4),
            on(Block::stepping(2, Some(2)), 1),
            on(Block::stepping(4, Some(2)), 8),
        ]
    }

    /// `microkernel` placed by `partitioning`, its widths settled.
    ///
    /// # Errors
    ///
    /// Why the microkernel cannot read the widths the problem admits.
    pub fn blueprint(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        microkernel: PlaneShuffleMicrokernel,
        partitioning: MatmulPartitioning,
    ) -> Result<PlaneShuffleMatmulRoutineBlueprint, Refusal> {
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint =
            MatmulBlueprint::new(hw, problem, microkernel, partitioning, vectorization, ());
        microkernel.check_vectors(&blueprint.widths)?;
        Ok(blueprint)
    }

    /// [`Routine::estimate`] against a float weight.
    fn float_estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &PlaneShuffleMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let view = MatmulView::of::<PlaneShuffleMicrokernel>(problem).expect(MatmulView::PREPARED);
        let (item, partitioning) = (view.work_item(), &blueprint.partitioning);
        let microkernel = &blueprint.microkernel;
        let (cube, leaf) = (partitioning.cube_box(), partitioning.leaf());
        let [rows, columns, k] = [M, N, K].map(|axis| cube.extent(axis) as u64);
        // Each run of columns a plane's units own reads every row of the lhs.
        let groups = columns / leaf.extent(N) as u64;
        let k_per_stage = leaf.extent(K) * microkernel.units_along_k;
        let stream = |operand: MatmulOperand, elements: u64, run: u64| {
            let size = item.elements.get(operand).size() as u64;
            GlobalStream {
                bytes: elements * size,
                run_bytes: (run * size) as usize,
            }
        };
        // A col weight's run is a column down all of `K`; a row weight's, one plane's columns.
        let rhs_run = match microkernel.orders.rhs {
            MatmulOrder::InBufferOrder => k,
            MatmulOrder::Stated => columns / grid.planes().count as u64,
        };
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * k) as f64,
            reads: vec![
                stream(MatmulOperand::Lhs, groups * rows * k, k),
                stream(MatmulOperand::Rhs, k * columns, rhs_run),
            ],
            writes: vec![stream(MatmulOperand::Out, rows * columns, columns)],
            stages: k.div_ceil(k_per_stage as u64),
            stages_in_flight: PlaneShuffleMicrokernel::STAGES_IN_FLIGHT as u32,
            drain_atomics: 0,
        });
        microkernel.estimate(&item, &(), grid.planes(), estimate);
    }

    /// [`Routine::estimate`] against `packing`, a packed weight.
    fn packed_estimate(
        &self,
        problem: &MatmulRoutineProblem,
        packing: &Packing,
        blueprint: &PlaneShuffleMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let view = MatmulView::of::<PlaneShuffleMicrokernel>(problem).expect(MatmulView::PREPARED);
        let (item, microkernel) = (view.work_item(), &blueprint.microkernel);
        let [rows, k] = [view.rows(), problem.k].map(|extent| extent as u64);
        // A step of the plane: one unit's block of `K`, over both its digits, by the units
        // sharing it.
        let leaf = blueprint.partitioning.leaf();
        let (blocks, inside) = blocked(K);
        let step_k = (leaf.extent(blocks) * leaf.extent(inside) * microkernel.units_along_k) as u64;
        // The columns of a cube's box, over both digits of `N`: the planes its order groups.
        let columns = blueprint.partitioning.columns();
        let groups = columns / microkernel.columns() as u64;
        let size = |operand: MatmulOperand| item.elements.get(operand).size() as u64;
        let rhs = packing.bytes(k as usize * columns as usize) as u64;
        estimate.staging = Some(CubeStaging {
            operations: 2.0 * (rows * columns * k) as f64,
            reads: vec![
                GlobalStream {
                    bytes: groups * rows * k * size(MatmulOperand::Lhs),
                    run_bytes: (k * size(MatmulOperand::Lhs)) as usize,
                },
                GlobalStream {
                    bytes: rhs,
                    run_bytes: (rhs / columns.max(1)) as usize,
                },
            ],
            writes: vec![GlobalStream {
                bytes: rows * columns * size(MatmulOperand::Out),
                run_bytes: (columns * size(MatmulOperand::Out)) as usize,
            }],
            stages: k.div_ceil(step_k),
            stages_in_flight: 1,
            drain_atomics: 0,
        });
        microkernel.estimate(&item, &(), grid.planes(), estimate);
    }

    /// [`Routine::prepare`] against a float weight.
    fn prepare_float(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &PlaneShuffleMatmulRoutineSettings,
    ) -> Result<(PlaneShuffleMatmulRoutineBlueprint, Grid), Refusal> {
        let view = MatmulView::of::<PlaneShuffleMicrokernel>(problem)?;
        let microkernel =
            PlaneShuffleMicrokernel::best(hw, &view.work_item(), &settings.microkernel)?;
        let partitioning = view.plane_partitioning(
            microkernel.tiling(),
            &[M, N],
            PlaneOrder::new(&[(N, settings.planes.get())], CubeOrder::RowMajor),
        );
        let blueprint = self.blueprint(hw, problem, microkernel, partitioning)?;
        let cube = CubeResources {
            plane_width: microkernel.plane_width() as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, blueprint.partitioning.as_cubek(), cube)?;
        Ok((blueprint, grid))
    }

    /// [`Routine::prepare`] against `packing`, a packed weight.
    fn prepare_packed(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        packing: &Packing,
        settings: &PlaneShuffleMatmulRoutineSettings,
    ) -> Result<(PlaneShuffleMatmulRoutineBlueprint, Grid), Refusal> {
        let PlaneShufflePackedPlacement {
            microkernel,
            columns,
            planes,
        } = PlaneShufflePackedPlacement::new(hw, problem, settings)?;
        let partitioning = MatmulView::of::<PlaneShuffleMicrokernel>(problem)?.plane_partitioning(
            microkernel.tiling(),
            &[columns],
            PlaneOrder::new(&[(columns, planes)], CubeOrder::RowMajor),
        );
        let cube = CubeResources {
            plane_width: microkernel.plane_width() as u32,
            shared_memory: 0,
        };
        let blueprint = MatmulBlueprint::packed(packing, microkernel, partitioning);
        let grid = Grid::new(hw, blueprint.partitioning.as_cubek(), cube)?;
        Ok((blueprint, grid))
    }
}

/// Where the plane gemv places a packed product: its microkernel, the axis its columns land on,
/// and the planes one cube holds side by side along it, a target snapped to what the runs of
/// columns divide, so no cube overhangs. The gemv deals the planes the launch's columns; the
/// decode into a float weight places the same planes in cubes, each cube copying its box.
pub struct PlaneShufflePackedPlacement {
    pub microkernel: PlaneShuffleMicrokernel,
    pub columns: Axis,
    pub planes: usize,
}

impl PlaneShufflePackedPlacement {
    /// The placement of `problem` at `settings` on `hw`.
    ///
    /// # Errors
    ///
    /// [`MATMUL_Q_RHS_NOT_PACKED`], for a float rhs; why `hw` cannot walk `problem`'s steps, or
    /// run its microkernel at `settings`.
    pub fn new(
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &PlaneShuffleMatmulRoutineSettings,
    ) -> Result<Self, Refusal> {
        if problem.packings.rhs.is_none() {
            return Err(MATMUL_Q_RHS_NOT_PACKED);
        }
        let view = MatmulView::of::<PlaneShuffleMicrokernel>(problem)?;
        view.check_steps(hw)?;
        let microkernel =
            PlaneShuffleMicrokernel::best(hw, &view.work_item(), &settings.microkernel)?;
        let (columns, _) = microkernel.column_cut();
        let runs = problem.n / (microkernel.columns() * microkernel.groups_along_n);
        let planes = divisor_at_most(
            runs,
            settings
                .planes
                .get()
                .min(hw.strip_budget(microkernel.plane_width())),
        );
        Ok(Self {
            microkernel,
            columns,
            planes,
        })
    }
}

impl Routine for PlaneShuffleMatmulRoutine {
    type Problem = MatmulRoutineProblem;
    type Microkernel = PlaneShuffleMicrokernel;
    type Settings = PlaneShuffleMatmulRoutineSettings;
    type Blueprint = PlaneShuffleMatmulRoutineBlueprint;
    type Partitioning = MatmulPartitioning;

    /// Its microkernel's blocks ([`PlaneShuffleMicrokernel::settings`]) at four planes a cube, and
    /// its own knob, planes a cube, at the default block: one and eight, since they trade cubes
    /// for the device's compute units against a cube's size, and last a cube as full as the device
    /// holds ([`Hardware::full_cube_planes`]). The default sits on the deepest step
    /// whose step of `K` still divides a decode projection's (2048, 6144) on a wave64 plane. The
    /// blocks a row weight wins with run on one to eight planes, and the deep block on one; a col
    /// weight declines every one of those. The default at two rows a unit runs on four planes.
    /// Against a packed weight it races [`packed_settings`](PlaneShuffleMatmulRoutine::packed_settings).
    ///
    /// The order is also what a caller naming the routine and no setting runs: the `PlaneShuffle` pin
    /// and the `Tile` policy launch the first setting that serves the product
    /// ([`launch_first_served_on`](MatmulRoutine::launch_first_served_on)), so a reorder changes what
    /// they run, not only the order a table measures in.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        use PlaneShuffleMicrokernelSettings as Block;
        let on = |microkernel, planes| Self::Settings {
            microkernel,
            planes: NonZeroUsize::new(planes).unwrap(),
        };
        vec![
            on(Block::DEFAULT, 4),
            on(Block::ONE_READ, 4),
            on(Block::TWO_COLUMNS, 4),
            on(Block::ONE_TRANSACTION, 4),
            on(Block::DEFAULT, 1),
            on(Block::DEFAULT, 8),
            on(Block::EIGHT_READS_ONE_TRANSACTION, 1),
            on(Block::ONE_READ_ONE_TRANSACTION, 4),
            on(Block::TWO_ROW_VECTORS, 1),
            on(Block::TWO_ROW_VECTORS, 4),
            on(Block::FOUR_ROW_VECTORS, 1),
            on(Block::FOUR_ROW_VECTORS, 4),
            on(Block::FOUR_ROW_VECTORS, 8),
            on(Block::TWO_ROWS_A_UNIT, 4),
            on(Block::DEFAULT, hw.full_cube_planes()),
        ]
    }

    /// The first of the settings for the weight `problem` states that prepares a launch on this
    /// device: [`settings`](Routine::settings) against a float weight,
    /// [`packed_settings`](PlaneShuffleMatmulRoutine::packed_settings) against a packed one.
    ///
    /// # Errors
    ///
    /// The last setting's decline, where none serves.
    fn prepare_first_served(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let settings = match problem.packings.rhs {
            Some(_) => self.packed_settings(),
            None => self.settings(hw),
        };
        let mut declined = ROUTINE_LISTS_NO_SETTING;
        for settings in settings {
            match self.prepare(hw, problem, &settings) {
                Ok(prepared) => return Ok(prepared),
                Err(why) => declined = why,
            }
        }
        Err(declined)
    }

    /// Nothing of its own: it walks the launch's planes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &PlaneShuffleMicrokernel,
        _: &MatmulPartitioning,
        vectorization: &mut Vectorization<MatmulOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Its units' steps: each group of units reads the lhs's rows along the whole of `K`, and the
    /// weight's columns it owns, straight from global memory, once: its values, or its words and
    /// scales where it is packed. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &MatmulRoutineProblem,
        blueprint: &PlaneShuffleMatmulRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        match &problem.packings.rhs {
            Some(packing) => self.packed_estimate(problem, packing, blueprint, grid, estimate),
            None => self.float_estimate(problem, blueprint, grid, estimate),
        }
    }

    /// Against a float weight, the launch's planes in cubes side by side along `n`; against a
    /// packed one, over its block-split space, as many planes a cube as divide the runs of columns
    /// its microkernel reads (`PlaneShufflePackedPlacement`).
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &MatmulRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        match &problem.packings.rhs {
            Some(packing) => self.prepare_packed(hw, problem, packing, settings),
            None => self.prepare_float(hw, problem, settings),
        }
    }
}

impl MatmulRoutine<PlaneShuffleMicrokernel> for PlaneShuffleMatmulRoutine {
    fn dispatch<O: DestinationLaunch, Out: MatmulOutput>(
        &self,
        tensors: &MatmulTensors<Out>,
        binding: MatmulBinding<O::Operand, TileArgLaunch, TileArgLaunch>,
        blueprint: PlaneShuffleMatmulRoutineBlueprint,
        grid: Grid,
    ) {
        let [sum, _, _, out] = tensors.work_item().kernel_elements();
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            plane_shuffle_matmul_kernel::launch_unchecked::<O>(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.widths.out,
                binding.lhs,
                binding.rhs,
                O::arg(binding.out),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel.tile_instruction(),
                [sum, out],
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::microkernel::MatmulLayout;
    use crate::microkernel::MatmulOperands;
    use crate::microkernel::dense::plane::PlaneShuffleKSplit;
    use crate::stored::Operand;
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::quant::scheme::{QuantScheme, QuantStore, QuantValue, ScaleDtype};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// Devices committing to a plane width, of 32 and of 64 units.
    fn committed() -> [Hardware; 5] {
        [
            devices::strix_halo(),
            devices::cdna_wave64(),
            devices::m2_pro(),
            devices::h100(),
            devices::cpu_12(),
        ]
    }

    /// A decode projection, `m` rows against a `2048 × 2048` weight laid as `weight`.
    fn projection(m: usize, rhs_layout: MatmulLayout) -> MatmulRoutineProblem {
        MatmulRoutineProblem::dense(m, 2048, 2048, F16, rhs_layout)
    }

    /// The default leads the list and no setting is listed twice.
    #[test]
    fn the_default_leads_and_no_setting_repeats() {
        let settings = PlaneShuffleMatmulRoutine.settings(&devices::strix_halo());
        assert_eq!(settings[0], PlaneShuffleMatmulRoutineSettings::DEFAULT);
        let names: std::collections::HashSet<_> =
            settings.iter().map(ToString::to_string).collect();
        assert_eq!(names.len(), settings.len());
    }

    /// Every setting prepares wherever the device commits to a plane width, and declines with its
    /// reason wherever it does not; a block of more than one column declines a col weight.
    #[test]
    fn a_setting_runs_where_the_device_commits_to_a_width() {
        let problem = projection(1, MatmulLayout::Col);
        for hw in committed()
            .into_iter()
            .chain([devices::wgpu_no_subgroups(), devices::rdna_range()])
        {
            for settings in PlaneShuffleMatmulRoutine.settings(&hw) {
                match (
                    PlaneShuffleMatmulRoutine.prepare(&hw, &problem, &settings),
                    hw.plane_size().is_ok(),
                ) {
                    (Ok(_), true) => {}
                    (Err(error), true) if settings.microkernel.columns.get() > 1 => {
                        assert_eq!(error, PlaneShuffleMicrokernel::COL_RHS_ONE_COLUMN_A_UNIT)
                    }
                    (Err(Refusal::Declined { why: details }), false) => {
                        assert_eq!(details, PlaneShuffleMicrokernel::DECLINE, "{settings}")
                    }
                    (prepared, _) => panic!("{settings} on {hw:?}: {prepared:?}"),
                }
            }
        }
    }

    /// A col weight is read down `K` beside the lhs, a line of each at a time, and written a
    /// column at a time; a row weight is read along `N` and written a line at a time, the lhs a
    /// line of the block's depth at a time.
    #[test]
    fn the_weight_is_read_along_the_axis_its_buffer_runs() {
        let hw = devices::strix_halo();
        let widths = |rhs_layout| {
            let (blueprint, _) = PlaneShuffleMatmulRoutine
                .prepare(
                    &hw,
                    &projection(1, rhs_layout),
                    &PlaneShuffleMatmulRoutineSettings::DEFAULT,
                )
                .expect("the default prepares on a committed wave32");
            let widths = blueprint.widths;
            (widths.lhs, widths.rhs, widths.out)
        };
        assert_eq!(widths(MatmulLayout::Col), (8, 8, 1));
        assert_eq!(widths(MatmulLayout::Row), (4, 8, 8));
    }

    /// A col weight whose `K` the plane's steps overhang would bind one element at a time, which
    /// the plane shuffle cannot read; a row weight of the same shape runs.
    #[test]
    fn a_col_weight_needs_vectors_along_k() {
        let hw = devices::strix_halo();
        let ragged = |rhs_layout| MatmulRoutineProblem::dense(1, 2048, 2040, F16, rhs_layout);
        let settings = PlaneShuffleMatmulRoutineSettings::DEFAULT;
        assert!(matches!(
            PlaneShuffleMatmulRoutine.prepare(&hw, &ragged(MatmulLayout::Col), &settings),
            Err(error) if error == PlaneShuffleMicrokernel::COL_RHS_NEEDS_VECTORS_ALONG_K
        ));
        assert!(
            PlaneShuffleMatmulRoutine
                .prepare(&hw, &ragged(MatmulLayout::Row), &settings)
                .is_ok()
        );
    }

    /// A cube holds every row of the product, so the weight is read once whatever the rows.
    #[test]
    fn a_cube_holds_every_row() {
        let hw = devices::strix_halo();
        let rows = |m| {
            let (blueprint, _) = PlaneShuffleMatmulRoutine
                .prepare(
                    &hw,
                    &projection(m, MatmulLayout::Col),
                    &PlaneShuffleMatmulRoutineSettings::DEFAULT,
                )
                .expect("the default prepares on a committed wave32");
            (
                blueprint.microkernel.rows,
                blueprint.partitioning.cube_box().extent(M),
            )
        };
        for m in 1..=PlaneShuffleMicrokernel::MAX_ROWS {
            assert_eq!(rows(m), (m, m));
        }
    }

    /// A product of no rows, or of more than a work item holds, is declined before anything is
    /// sized to it.
    #[test]
    fn the_rows_a_work_item_holds_bound_the_product() {
        let settings = PlaneShuffleMatmulRoutineSettings::DEFAULT;
        for m in [0, PlaneShuffleMicrokernel::MAX_ROWS + 1] {
            assert!(matches!(
                PlaneShuffleMatmulRoutine.prepare(&devices::strix_halo(), &projection(m, MatmulLayout::Col), &settings),
                Err(error) if error == PlaneShuffleMicrokernel::ROWS_OUT_OF_RANGE
            ));
        }
    }

    /// The whole plane reads `K` at whatever width the device runs, and launches that wide.
    #[test]
    fn the_whole_plane_is_the_devices_width() {
        let problem = projection(1, MatmulLayout::Col);
        let settings = PlaneShuffleMatmulRoutineSettings::DEFAULT;
        let prepare = |hw: Hardware| {
            PlaneShuffleMatmulRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the default prepares on a committed width")
        };
        let (wave32, wave32_grid) = prepare(devices::strix_halo());
        let (wave64, wave64_grid) = prepare(devices::cdna_wave64());
        assert_eq!(wave32.microkernel.units_along_k, 32);
        assert_eq!(wave64.microkernel.units_along_k, 64);
        assert_eq!(wave32_grid.planes().width, 32);
        assert_eq!(wave64_grid.planes().width, 64);
    }

    /// A transaction's worth of units along `K` is as many col steps as one transaction holds:
    /// fewer for a deeper block, the same at every element, since a block counts reads; and one
    /// for a row weight, whose units beside each other fill it.
    #[test]
    fn a_transactions_worth_of_units_moves_with_the_depth() {
        let hw = devices::strix_halo();
        let units = |element, depth, rhs_layout| {
            let item = MatmulView::of::<PlaneShuffleMicrokernel>(&MatmulRoutineProblem::dense(
                1, 2048, 2048, element, rhs_layout,
            ))
            .expect("a weight lined along an axis")
            .work_item();
            let plane_width = 32;
            PlaneShuffleKSplit::OneTransaction.units(&hw, &item, depth, plane_width)
        };
        let reads = |count| NonZeroUsize::new(count).unwrap();
        let f32 = ElemType::Float(FloatKind::F32);
        assert_eq!(units(F16, reads(1), MatmulLayout::Col), 8);
        assert_eq!(units(f32, reads(1), MatmulLayout::Col), 8);
        assert_eq!(units(F16, reads(2), MatmulLayout::Col), 4);
        assert_eq!(units(F16, reads(1), MatmulLayout::Row), 1);
    }

    /// `m` rows against a `k × n` weight of `value` whose words run `along`, under `f16` block
    /// scales of 32 along the same axis.
    fn packed(
        m: usize,
        n: usize,
        k: usize,
        along: Axis,
        value: QuantValue,
    ) -> MatmulRoutineProblem {
        let (block, store) = match along == K {
            true => (vec![32, 1], QuantStore::PackedU32(1)),
            false => (vec![32], QuantStore::PackedU32(0)),
        };
        let scheme = QuantScheme::default()
            .with_value(value)
            .per_block(block, ScaleDtype::F16)
            .with_store(store);
        let packing = Packing {
            operand: Operand::of_scheme(&scheme, along, [K, N]).expect("a packed weight"),
            scheme,
        };
        MatmulRoutineProblem::packed(
            m,
            n,
            k,
            F16,
            MatmulOperands {
                rhs: Some(packing),
                ..MatmulOperands::all(None)
            },
        )
    }

    /// A q4 decode projection, `m` rows against a packed `4096 × 4096` weight read along `k`.
    fn packed_projection(m: usize) -> MatmulRoutineProblem {
        packed(m, 4096, 4096, K, QuantValue::Q4S)
    }

    /// The packed lead leads the packed settings and no setting is listed twice.
    #[test]
    fn the_packed_lead_leads_and_no_packed_setting_repeats() {
        let settings = PlaneShuffleMatmulRoutine.packed_settings();
        assert_eq!(settings[0], PlaneShuffleMatmulRoutineSettings::PACKED);
        let names: std::collections::HashSet<_> =
            settings.iter().map(ToString::to_string).collect();
        assert_eq!(names.len(), settings.len());
    }

    /// Every setting prepares wherever the device commits to a plane width, and declines with its
    /// reason wherever it does not.
    #[test]
    fn a_setting_runs_on_a_packed_weight_where_the_device_commits_to_a_width() {
        let problem = packed_projection(1);
        for hw in committed()
            .into_iter()
            .chain([devices::wgpu_no_subgroups(), devices::rdna_range()])
        {
            for settings in PlaneShuffleMatmulRoutine.packed_settings() {
                match (
                    PlaneShuffleMatmulRoutine.prepare(&hw, &problem, &settings),
                    hw.plane_size().is_ok(),
                ) {
                    (Ok(_), true) => {}
                    (Err(Refusal::Declined { why: details }), false) => {
                        assert_eq!(details, PlaneShuffleMicrokernel::DECLINE, "{settings}")
                    }
                    (prepared, _) => panic!("{settings} on {hw:?}: {prepared:?}"),
                }
            }
        }
    }

    /// A cube's box, the planes its order groups, holds `n` over the cubes in columns: what the
    /// estimate charges a cube, counted from the planes and not from the grid. The vocabulary-wide
    /// head spills past one grid dimension; a spare cube there would run past the last column,
    /// an edge a vectorized output cannot mask.
    #[test]
    fn a_cubes_columns_are_its_share_of_a_packed_launch() {
        let problems = [
            packed_projection(1),
            packed(1, 151_936, 1024, K, QuantValue::Q4S),
            packed(1, 4096, 4096, N, QuantValue::Q4S),
            packed(3, 1024, 2048, K, QuantValue::Q8S),
            packed(3, 96, 1024, K, QuantValue::Q4S),
        ];
        for hw in committed() {
            for problem in &problems {
                for settings in PlaneShuffleMatmulRoutine.packed_settings() {
                    let (blueprint, grid) = PlaneShuffleMatmulRoutine
                        .prepare(&hw, problem, &settings)
                        .expect("a committed width places the projection");
                    let cubes = grid.cubes().expect("a static grid");
                    assert_eq!(
                        blueprint.partitioning.columns() * cubes,
                        problem.n as u64,
                        "{settings} on {hw:?}"
                    );
                }
            }
        }
    }

    /// A product of no rows, or of more than a work item holds, is declined before anything is
    /// sized to it; a two-level weight prepares as a one-level one does.
    #[test]
    fn the_rows_a_work_item_holds_bound_a_packed_product() {
        let hw = devices::strix_halo();
        let settings = PlaneShuffleMatmulRoutineSettings::PACKED;
        for m in [0, PlaneShuffleMicrokernel::MAX_ROWS + 1] {
            assert_eq!(
                PlaneShuffleMatmulRoutine
                    .prepare(&hw, &packed_projection(m), &settings)
                    .err(),
                Some(PlaneShuffleMicrokernel::ROWS_OUT_OF_RANGE)
            );
        }
        assert!(
            PlaneShuffleMatmulRoutine
                .prepare(
                    &hw,
                    &packed_projection(PlaneShuffleMicrokernel::MAX_ROWS),
                    &settings
                )
                .is_ok()
        );
    }
}
