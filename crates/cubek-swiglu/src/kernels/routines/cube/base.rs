use super::kernel::cube_swiglu_kernel;
use crate::kernels::{
    SwigluBinding, SwigluBlueprint, SwigluRoutine, SwigluRoutineProblem, SwigluTensors,
};
use core::num::NonZeroUsize;
use cubecl::tile::Levels;
use cubecl::tile::launch::DestinationLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::row_store::OwnerMemory;
use cubek_microkernel::swiglu::cube::{CubeSwigluMicrokernel, CubeSwigluMicrokernelSettings};
use cubek_microkernel::swiglu::{H, R, SwigluOperand, SwigluPartitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// The gate with every run of a row taken by a group of units, a cube holding several groups side
/// by side, each on a row of its own. Into a plain output it asks nothing of the hardware, so it
/// runs everywhere: this kind's floor. Into a packed one it needs plane instructions, and its runs
/// are whole words of block scales.
///
/// Each group reads its run of either half where it lies, its units in turns along it, and writes
/// the run's lines; no unit waits on another.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeSwigluRoutine;

/// What a [`CubeSwigluRoutine`] launch is measured at: its microkernel's group and lines, and how
/// many of its groups a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeSwigluRoutineSettings {
    pub microkernel: CubeSwigluMicrokernelSettings,
    /// Groups a cube holds, a row each: a target, fewer where the cube's units run out, more where
    /// the rows would take more cubes than the device spawns along the grid's `x`.
    pub groups: NonZeroUsize,
}

impl CubeSwigluRoutineSettings {
    /// One group of up to 256 units a cube, four lines each: a decode step's row read by a few
    /// cubes, every unit with several loads in flight.
    pub const DEFAULT: Self = Self {
        microkernel: CubeSwigluMicrokernelSettings::DEFAULT,
        groups: NonZeroUsize::MIN,
    };

    /// The microkernel's settings on `hw` for `problem`'s rows: the settings' own, its group
    /// narrowed where groups of its width would leave a cube too few of them for the rows' cubes
    /// to fit the grid's `x`, so the floor's grid fits any rows a grid of one-unit groups holds.
    fn microkernel_for(
        &self,
        hw: &Hardware,
        problem: &SwigluRoutineProblem,
    ) -> CubeSwigluMicrokernelSettings {
        let hardware = &hw.properties.hardware;
        let units = (hardware.max_units_per_cube as usize / Self::groups_to_fit(hw, problem))
            .max(1)
            .min(self.microkernel.units.get());
        CubeSwigluMicrokernelSettings {
            units: NonZeroUsize::new(units).unwrap(),
            ..self.microkernel
        }
    }

    /// Groups a cube of `microkernel`'s groups holds on `hw` for `problem`: the target, at least
    /// as many as keep the rows' cubes within the grid's `x`, and as many as the cube's units allow
    /// and no more than `staged`, the owners its store's stage fits. A problem whose rows outgrow
    /// a grid of full cubes is past any launch of this kind.
    fn groups_of(
        &self,
        hw: &Hardware,
        microkernel: &CubeSwigluMicrokernel,
        problem: &SwigluRoutineProblem,
        staged: usize,
    ) -> usize {
        let hardware = &hw.properties.hardware;
        self.groups
            .get()
            .max(Self::groups_to_fit(hw, problem))
            .min((hardware.max_units_per_cube as usize / microkernel.units).max(1))
            .min(staged)
    }

    /// The fewest groups a cube holds for `problem`'s rows to take no more cubes than `hw`'s grid
    /// spawns along its `x`.
    fn groups_to_fit(hw: &Hardware, problem: &SwigluRoutineProblem) -> usize {
        let cubes = hw.properties.hardware.max_cube_count.0.max(1) as usize;
        problem.rows.div_ceil(cubes)
    }
}

impl core::fmt::Display for CubeSwigluRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cube-swiglu {} groups={}", self.microkernel, self.groups)
    }
}

pub type CubeSwigluRoutineBlueprint = SwigluBlueprint<CubeSwigluMicrokernel>;

impl Routine for CubeSwigluRoutine {
    type Problem = SwigluRoutineProblem;
    type Microkernel = CubeSwigluMicrokernel;
    type Settings = CubeSwigluRoutineSettings;
    type Blueprint = CubeSwigluRoutineBlueprint;
    type Partitioning = SwigluPartitioning;

    /// Each of its microkernel's settings ([`CubeSwigluMicrokernel::settings`]) a group a cube,
    /// then the smaller groups several a cube: a prompt's many rows, more of them a cube.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let on = |microkernel, groups| Self::Settings {
            microkernel,
            groups: NonZeroUsize::new(groups).unwrap(),
        };
        CubeSwigluMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| on(microkernel, 1))
            .chain([
                on(CubeSwigluMicrokernelSettings::SHORT_RUNS, 4),
                on(CubeSwigluMicrokernelSettings::LONG_RUNS, 4),
            ])
            .collect()
    }

    /// Nothing of its own: it walks cubes and groups, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeSwigluMicrokernel,
        _: &SwigluPartitioning,
        vectorization: &mut Vectorization<SwigluOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each cube reads its groups' runs of either half once, straight from global memory, and
    /// writes them once. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &SwigluRoutineProblem,
        blueprint: &CubeSwigluRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        estimate.staging = Some(problem.cube_staging(&blueprint.partitioning.cube_box()));
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }

    /// A group is launched as the cube's `x`, its units the group's, and the groups along `y`: the
    /// units share nothing, so the width a plane runs at is not read.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &SwigluRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel = CubeSwigluMicrokernel::best(
            hw,
            &problem.work_item(),
            &settings.microkernel_for(hw, problem),
        )?;
        let store = problem.store;
        store.runs_on(hw)?;
        let staged = store.owners_staged(
            hw,
            OwnerMemory {
                staged: microkernel.run(),
                element: problem.element,
                scratch_bytes: 0,
            },
        );
        let groups = settings.groups_of(hw, &microkernel, problem, staged);
        let run_box = [vec![(R, 1), (H, 2)], store.width_axes(microkernel.run())].concat();
        let partitioning = SwigluPartitioning::new(
            problem.space(),
            Levels::leaf(&run_box).planes(&[(R, groups)]),
            store,
        );
        let cube = CubeResources {
            plane_width: microkernel.units as u32,
            shared_memory: store.stage_bytes(groups * microkernel.run(), problem.element),
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint = SwigluBlueprint::new(hw, problem, microkernel, partitioning, vectorization)
            .planned_at(microkernel.width)?;
        Ok((blueprint, grid))
    }
}

impl SwigluRoutine<CubeSwigluMicrokernel> for CubeSwigluRoutine {
    fn dispatch<O: DestinationLaunch, Out>(
        &self,
        tensors: &SwigluTensors<Out>,
        binding: SwigluBinding<O::Operand>,
        blueprint: CubeSwigluRoutineBlueprint,
        grid: Grid,
    ) {
        // SAFETY: every operand is bound through the launcher, and the output through `O`, which
        // mask the reads and writes of a tile the problem's edge cuts short, and the kernel
        // indexes no buffer but through its tiles; its shared memory is a packed store's stage,
        // each group writing its own run, encoded only once the cube has synced.
        unsafe {
            cube_swiglu_kernel::launch_unchecked::<O>(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.width,
                binding.input.arg(),
                O::arg(binding.out),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                blueprint.stage,
                tensors.problem().element,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// The floor prepares every setting of every problem on every device, plane instructions or
    /// none, a committed width or a range.
    #[test]
    fn the_floor_prepares_everywhere() {
        for hw in [
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::strix_halo(),
            devices::h100(),
            devices::cpu_12(),
        ] {
            for settings in CubeSwigluRoutine.settings(&hw) {
                for problem in problems() {
                    let prepared = CubeSwigluRoutine.prepare(&hw, &problem, &settings);
                    assert!(prepared.is_ok(), "{settings} on {hw:?}: {prepared:?}");
                }
            }
        }
    }

    /// A run is the longest that divides the half within the group's lines, so no tile overhangs
    /// it, and a group takes no more units than its run has lines: a short half's run is the half,
    /// a prime half binds one element wide, so its lines are its elements, and a half whose lines
    /// have no divisor that fills the group is one run, its units taking more lines.
    #[test]
    fn a_run_divides_its_row() {
        let hw = devices::strix_halo();
        let microkernel = |dim: usize, element| {
            let problem = SwigluRoutineProblem::dense(1, dim, element);
            let (blueprint, _) = CubeSwigluRoutine
                .prepare(&hw, &problem, &CubeSwigluRoutineSettings::DEFAULT)
                .expect("the floor prepares");
            let microkernel = blueprint.microkernel;
            (microkernel.run_lines, microkernel.units, blueprint.width)
        };
        assert_eq!(microkernel(8192, F16), (1024, 256, 8));
        assert_eq!(microkernel(128, F16), (16, 16, 8));
        assert_eq!(microkernel(2816, F16), (352, 256, 8));
        assert_eq!(microkernel(111, F32), (111, 64, 1));
        assert_eq!(microkernel(1031 * 2, F32), (1031, 256, 2));
        assert_eq!(microkernel(5003, F32), (5003, 256, 1));
    }

    /// A row longer than a group unrolls is walked in a loop: one run, every unit more lines than
    /// it unrolls, rather than runs of a line in cubes of a unit.
    #[test]
    fn a_long_prime_row_is_one_run_walked_in_a_loop() {
        let hw = devices::strix_halo();
        let problem = SwigluRoutineProblem::dense(1, 5003, F32);
        let (blueprint, _) = CubeSwigluRoutine
            .prepare(&hw, &problem, &CubeSwigluRoutineSettings::DEFAULT)
            .expect("the floor prepares");
        assert!(
            !blueprint.microkernel.unrolled(),
            "{:?}",
            blueprint.microkernel
        );
    }

    /// Rows past the cubes a grid's `x` spawns ride more groups a cube, whatever the setting
    /// targets: wgpu's `x` holds 65535 cubes, CUDA's `2³¹ - 1`.
    #[test]
    fn rows_past_the_grid_take_more_groups_a_cube() {
        let problem = SwigluRoutineProblem::dense(70_000, 128, F16);
        let groups = |hw: &Hardware| {
            let (_, grid) = CubeSwigluRoutine
                .prepare(hw, &problem, &CubeSwigluRoutineSettings::DEFAULT)
                .expect("the floor prepares");
            grid.planes().count
        };
        assert_eq!(groups(&devices::wgpu_no_subgroups()), 2);
        assert_eq!(groups(&devices::h100()), 1);
    }

    /// A prompt's rows past the cubes a grid's `x` spawns, on a row wide enough to fill a cube's
    /// units with one group: the floor narrows its groups until enough of them fit a cube, so it
    /// still prepares, within the grid.
    #[test]
    fn rows_past_the_grid_of_a_wide_row_narrow_the_floors_groups() {
        let hw = devices::wgpu_no_subgroups();
        let problem = SwigluRoutineProblem::dense(70_000, 8192, F16);
        let (_, grid) = CubeSwigluRoutine
            .prepare(&hw, &problem, &CubeSwigluRoutineSettings::DEFAULT)
            .expect("the floor prepares past the grid's x");
        let cubes = grid.launch.cube_count;
        let most = hw.properties.hardware.max_cube_count;
        assert!(
            matches!(cubes, cubecl::CubeCount::Static(x, _, _) if x <= most.0),
            "{cubes:?} within {most:?}"
        );
    }

    /// Two settings that derive one launch prepare one blueprint, so a round keeps only the first:
    /// on a CPU's few units every 64- and 256-unit group is a group of its units.
    #[test]
    fn settings_deriving_one_launch_prepare_one_blueprint() {
        let hw = devices::cpu_12();
        let problem = SwigluRoutineProblem::dense(1, 8192, F32);
        let blueprint = |microkernel| {
            let settings = CubeSwigluRoutineSettings {
                microkernel,
                groups: NonZeroUsize::MIN,
            };
            CubeSwigluRoutine
                .prepare(&hw, &problem, &settings)
                .expect("the floor prepares")
                .0
        };
        assert_eq!(
            blueprint(CubeSwigluMicrokernelSettings::DEFAULT),
            blueprint(CubeSwigluMicrokernelSettings::SHORT_RUNS)
        );
    }

    /// The up half lies a half into its row, so a half whose width the widest read does not
    /// divide binds narrower, at the width that divides it, for both operands alike.
    #[test]
    fn the_up_half_binds_at_the_width_its_start_allows() {
        let hw = devices::strix_halo();
        let problem = SwigluRoutineProblem::dense(4, 1028, F16);
        let (blueprint, _) = CubeSwigluRoutine
            .prepare(&hw, &problem, &CubeSwigluRoutineSettings::DEFAULT)
            .expect("the floor prepares");
        assert_eq!(blueprint.width, 4);
    }

    /// Under a packed store a run is whole words of block scales, and a cube holds no more groups
    /// than its stage fits in the device's budget: a decode step's row stages in runs, and a
    /// prompt's rows take fewer groups a cube than the setting targets.
    #[test]
    fn a_packed_store_runs_whole_words_of_scales_within_its_stage() {
        use cubek_microkernel::row_store::RowStore;
        let store = RowStore::Packed {
            block: 32,
            scales_per_word: 4,
        };
        let hw = devices::rdna_range();
        for settings in CubeSwigluRoutine.settings(&hw) {
            for problem in [
                SwigluRoutineProblem::dense(1, 8192, F32).stored(store),
                SwigluRoutineProblem::dense(512, 2816, F16).stored(store),
            ] {
                let (blueprint, grid) = CubeSwigluRoutine
                    .prepare(&hw, &problem, &settings)
                    .unwrap_or_else(|why| panic!("{settings}: {why:?}"));
                assert!(
                    store.holds_cube_run(blueprint.microkernel.run()),
                    "{settings}"
                );
                let stage =
                    grid.planes().count * blueprint.microkernel.run() * problem.element.size();
                assert!(
                    stage <= hw.shared_memory_budget(),
                    "{settings}: {stage} bytes staged"
                );
            }
        }
    }

    /// A decode step's `d_ff`, a prompt's rows, a prime half that binds one element at a time, a
    /// half one line past a power of two, a prime half longer than a group unrolls, and more rows
    /// than a grid's `x` holds cubes on wgpu and CUDA.
    fn problems() -> [SwigluRoutineProblem; 6] {
        [
            SwigluRoutineProblem::dense(1, 8192, F16),
            SwigluRoutineProblem::dense(512 * 4, 2816, F16),
            SwigluRoutineProblem::dense(3, 111, F32),
            SwigluRoutineProblem::dense(2, 1028, F32),
            SwigluRoutineProblem::dense(1, 5003, F32),
            SwigluRoutineProblem::dense(70_000, 128, F16),
        ]
    }
}
