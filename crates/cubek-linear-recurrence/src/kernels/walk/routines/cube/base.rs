use super::delta::CubeDeltaWalkRoutine;
use crate::kernels::walk::{
    RECURRENCE_WALK_ELEMENT, RecurrenceWalkBlueprint, RecurrenceWalkProblem, settled_width,
};
use core::num::NonZeroUsize;
use cubecl::tile::Levels;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubek_microkernel::linear_recurrence::walk::RecurrenceWalkOperand;
use cubek_microkernel::linear_recurrence::walk::cube::{
    CubeDeltaWalkMicrokernel, CubeWalkMicrokernelSettings, MAX_HELD_LINES_PER_UNIT,
};
use cubek_microkernel::linear_recurrence::{
    HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION, RATE, RecurrenceWalkPartitioning, VALUE_FEATURE,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubeStaging, GlobalStream, KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// What a [`CubeDeltaWalkRoutine`] launch is measured at: its microkernel's cells, and how
/// many of its runs a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeDeltaWalkRoutineSettings {
    pub microkernel: CubeWalkMicrokernelSettings,
    /// Runs a cube holds, `columns` of the head's value columns each: a target, fewer where the
    /// cube's units run out or the head's runs of columns do not divide.
    pub runs: NonZeroUsize,
}

impl CubeDeltaWalkRoutineSettings {
    /// Eight runs a cube: a 128-wide head read four columns at a time in four cubes.
    pub const DEFAULT: Self = Self::new(CubeWalkMicrokernelSettings::DEFAULT, 8);
    /// One run a cube at the fewest registers: what every admitted problem prepares.
    pub const FLOOR: Self = Self::new(CubeWalkMicrokernelSettings::FLOOR, 1);

    pub const fn new(microkernel: CubeWalkMicrokernelSettings, runs: usize) -> Self {
        Self {
            microkernel,
            runs: NonZeroUsize::new(runs).unwrap(),
        }
    }

    /// Runs a cube holds on `hw` for `problem` at `microkernel`: the largest count no larger than
    /// the target that divides the head's runs of columns, and that the cube's units, its height
    /// and its shared memory hold.
    fn runs_of(
        &self,
        hw: &Hardware,
        microkernel: &CubeDeltaWalkMicrokernel,
        problem: &RecurrenceWalkProblem,
    ) -> usize {
        let in_head = problem.heads.value_features / microkernel.shape.columns();
        let in_memory = hw.shared_memory_budget() / microkernel.shape.shared_memory(1);
        let most = hw
            .strip_budget(microkernel.shape.units())
            .min(in_memory)
            .min(self.runs.get());
        (1..=most.min(in_head))
            .rev()
            .find(|runs| in_head.is_multiple_of(*runs))
            .unwrap_or(1)
    }
}

impl core::fmt::Display for CubeDeltaWalkRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cube-delta-walk {} runs={}", self.microkernel, self.runs)
    }
}

pub type CubeDeltaWalkRoutineBlueprint = RecurrenceWalkBlueprint<CubeDeltaWalkMicrokernel>;

impl Routine for CubeDeltaWalkRoutine {
    type Problem = RecurrenceWalkProblem;
    type Microkernel = CubeDeltaWalkMicrokernel;
    type Settings = CubeDeltaWalkRoutineSettings;
    type Blueprint = CubeDeltaWalkRoutineBlueprint;
    type Partitioning = RecurrenceWalkPartitioning;

    /// Each of its microkernel's settings at the default runs, then the default's cells in
    /// cubes of a whole head and of a few runs.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        let default = Self::Settings::DEFAULT;
        CubeDeltaWalkMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings::new(microkernel, default.runs.get()))
            .chain([
                Self::Settings::new(default.microkernel, 32),
                Self::Settings::new(default.microkernel, 2),
            ])
            .collect()
    }

    /// Nothing of its own: it walks cubes and runs, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeDeltaWalkMicrokernel,
        _: &RecurrenceWalkPartitioning,
        vectorization: &mut Vectorization<RecurrenceWalkOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each cube reads its head's queries and keys and its runs' values once a position, its runs'
    /// state once, and writes their outputs and kept states once. Then its microkernel, for the
    /// work.
    fn estimate(
        &self,
        problem: &RecurrenceWalkProblem,
        blueprint: &CubeDeltaWalkRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let input_bytes = RECURRENCE_WALK_ELEMENT.size();
        let columns = blueprint.runs * blueprint.microkernel.shape.columns();
        let positions = problem.positions;
        let key_features = problem.heads.key_features;
        let stream_at = |element_bytes: usize, elements: usize, run: usize| GlobalStream {
            bytes: (elements * element_bytes) as u64,
            run_bytes: run * element_bytes,
        };
        let stream = |elements: usize, run: usize| stream_at(input_bytes, elements, run);
        let state = key_features * columns;
        estimate.staging = Some(CubeStaging {
            operations: (7 * positions * state) as f64,
            reads: vec![
                stream(2 * positions * key_features, key_features),
                stream(positions * columns, columns),
                stream(2 * positions, 1),
                stream(state, columns),
            ],
            writes: vec![
                stream_at(problem.output_element.size(), positions * columns, columns),
                stream(problem.state.kept().get() * state, columns),
            ],
            stages: positions as u64,
            stages_in_flight: 1,
            drain_atomics: 0,
        });
        blueprint
            .microkernel
            .estimate(&problem.heads, &(), grid.planes(), estimate);
    }

    /// A run is launched as the cube's `x`, its units the run's, and the runs along `y`: the
    /// units meet in shared memory, so the width a plane runs at is not read.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &RecurrenceWalkProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        problem.admit_launch(hw)?;
        let heads = &problem.heads;
        // The width the operands bind at does not depend on the rows a unit holds, so it is settled
        // first, and the cells a unit holds are capped to the lines its registers bound at it.
        let mut vectorization = Vectorization::new();
        RecurrenceWalkOperand::require(heads.columns(), heads.columns(), &mut vectorization);
        let width = settled_width(hw, problem, vectorization);
        let microkernel = CubeDeltaWalkMicrokernel::best(
            hw,
            heads,
            &settings
                .microkernel
                .capped_at(MAX_HELD_LINES_PER_UNIT * width),
        )?;
        let runs = settings.runs_of(hw, &microkernel, problem);
        let mut unit_axes = vec![
            (POSITION, 1),
            (RATE, 2),
            (KEY_HEAD, 1),
            (HEAD_IN_GROUP, 1),
            (KEY_FEATURE, microkernel.shape.rows()),
            (VALUE_FEATURE, microkernel.shape.columns()),
        ];
        unit_axes.extend(problem.state.kept_axis());
        let unit = Levels::leaf(&unit_axes);
        let partitioning = RecurrenceWalkPartitioning::new(
            problem.space(),
            unit.walk_every(&[POSITION])
                .units(&[(KEY_FEATURE, microkernel.shape.units())])
                .planes(&[(VALUE_FEATURE, runs)]),
        );
        let blueprint = RecurrenceWalkBlueprint::new(microkernel, partitioning, width, runs);
        let cube = CubeResources {
            plane_width: microkernel.shape.units() as u32,
            shared_memory: microkernel.shape.shared_memory(runs),
        };
        let grid = Grid::new(hw, blueprint.partitioning.as_cubek(), cube)?;
        Ok((blueprint, grid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::walk::RecurrenceWalkStateLayout;
    use crate::kernels::walk::problem::fixtures::binding_enough;
    use cubecl::tile::launch::Refusal;
    use cubek_microkernel::linear_recurrence::RecurrenceHeads;
    use cubek_std::devices;
    use cubek_std::routine::EstimatedProblem;

    fn walk(
        batch: usize,
        positions: usize,
        heads: (usize, usize, usize, usize),
        kept: usize,
    ) -> RecurrenceWalkProblem {
        let (key_heads, heads_per_group, key_features, value_features) = heads;
        let heads = RecurrenceHeads {
            key_heads,
            heads_per_group,
            key_features,
            value_features,
        };
        RecurrenceWalkProblem::dense(
            batch,
            positions,
            heads,
            RECURRENCE_WALK_ELEMENT,
            RecurrenceWalkStateLayout::Owned {
                kept: NonZeroUsize::new(kept).expect("a state is kept"),
            },
        )
    }

    fn in_slots(
        batch: usize,
        positions: usize,
        heads: (usize, usize, usize, usize),
        slots: usize,
    ) -> RecurrenceWalkProblem {
        let mut problem = walk(batch, positions, heads, 1);
        problem.state = RecurrenceWalkStateLayout::Slots {
            slots: NonZeroUsize::new(slots).expect("a store holds a slot"),
        };
        problem
    }

    /// A decode step of a many-headed layer, a prompt of it, a prime position count with every
    /// state kept, value heads that are their own key heads, heads of odd widths, key rows no unit
    /// takes whole, and many sequences; then the walks in a store's slots of a decode step, three
    /// positions and an odd head.
    fn problems() -> [RecurrenceWalkProblem; 10] {
        [
            walk(1, 1, (16, 3, 128, 128), 1),
            walk(1, 512, (16, 3, 128, 128), 4),
            walk(2, 7, (2, 2, 64, 64), 7),
            walk(3, 5, (4, 1, 32, 32), 2),
            walk(1, 3, (1, 2, 16, 37), 1),
            walk(2, 4, (3, 1, 100, 12), 2),
            walk(64, 1, (16, 3, 128, 128), 1),
            in_slots(8, 1, (16, 3, 128, 128), 32),
            in_slots(2, 3, (4, 3, 128, 128), 5),
            in_slots(3, 2, (1, 2, 16, 37), 3),
        ]
    }

    fn fleet() -> [Hardware; 6] {
        [
            binding_enough(devices::h100()),
            binding_enough(devices::cdna_wave64()),
            binding_enough(devices::cpu_12()),
            binding_enough(devices::strix_halo()),
            binding_enough(devices::rdna_range()),
            binding_enough(devices::wgpu_no_subgroups()),
        ]
    }

    /// The floor prepares every problem of the GPUs of the fleet whose key rows its power-of-two
    /// units hold in whole tiles or masked ones, a committed width or a range, plane instructions
    /// or none; a cube of twelve units holds a 128-row head's floor in no units, and a head of
    /// 100 rows, whose tiles would leave a unit wholly past it unmasked, is held by none. Admission
    /// says so with the microkernel's refusal.
    #[test]
    fn the_floor_prepares_everything_a_cube_of_units_holds() {
        let floor = CubeDeltaWalkRoutineSettings::FLOOR;
        let declined = Refusal::Declined {
            why: CubeDeltaWalkMicrokernel::DECLINE,
        };
        for hw in fleet() {
            let cpu = hw.properties.hardware.max_units_per_cube < 256;
            for problem in problems() {
                let prepared = CubeDeltaWalkRoutine.prepare(&hw, &problem, &floor);
                let key_features = problem.heads.key_features;
                if key_features == 100 || (cpu && key_features == 128) {
                    assert_eq!(prepared.as_ref().err(), Some(&declined), "{problem:?}");
                } else if !cpu {
                    assert!(prepared.is_ok(), "{problem:?}: {prepared:?}");
                }
                assert_eq!(problem.admit(&hw), prepared.map(|_| ()));
            }
        }
    }

    /// Past admission no listed setting declines, aligned or off a whole read, and none holds more
    /// lines than the bound: the cells and the runs are targets, capped to what the registers, the
    /// cube and the shared memory hold.
    #[test]
    fn no_setting_past_admission_declines_or_holds_more_lines_than_the_bound() {
        for hw in fleet() {
            for settings in CubeDeltaWalkRoutine.settings(&hw) {
                for mut problem in problems() {
                    for start in [0, 4] {
                        problem.start_bytes.state = start;
                        if problem.admit(&hw).is_err() {
                            continue;
                        }
                        let (blueprint, _) = CubeDeltaWalkRoutine
                            .prepare(&hw, &problem, &settings)
                            .unwrap_or_else(|why| panic!("{settings} {problem:?}: {why:?}"));
                        let held = blueprint.microkernel.shape.held_lines(blueprint.width);
                        assert!(held <= MAX_HELD_LINES_PER_UNIT, "{settings}: {held}");
                    }
                }
            }
        }
    }

    /// A 128-wide head reads four columns at a time, its units holding the cells the settings
    /// target; heads of odd widths read a column at a time; key rows of three lines are held by
    /// one unit, and a head no power of two of tiles divides by eight units whose last are
    /// partial.
    #[test]
    fn a_unit_holds_the_cells_its_settings_target() {
        let hw = binding_enough(devices::strix_halo());
        let shape = |heads: (usize, usize, usize, usize), cells: usize| {
            let settings =
                CubeDeltaWalkRoutineSettings::new(CubeWalkMicrokernelSettings::new(cells), 8);
            let (blueprint, _) = CubeDeltaWalkRoutine
                .prepare(&hw, &walk(1, 7, heads, 1), &settings)
                .expect("the floor prepares");
            let microkernel = blueprint.microkernel;
            (
                microkernel.shape.units(),
                microkernel.shape.rows(),
                microkernel.shape.columns(),
            )
        };
        assert_eq!(shape((2, 3, 128, 128), 64), (8, 16, 4));
        assert_eq!(shape((2, 3, 128, 128), 32), (16, 8, 4));
        assert_eq!(shape((2, 3, 128, 37), 64), (8, 16, 1));
        assert_eq!(shape((2, 3, 12, 8), 64), (1, 12, 4));
        assert_eq!(shape((2, 3, 100, 12), 64), (8, 16, 4));
    }

    /// The partitioning states what a unit owns: its leaf holds a unit's rows, its units level cuts
    /// the key features into one leaf for each of the microkernel's units, and the tile engine
    /// masks the rows the last units overhang exactly where the units reach past the head.
    #[test]
    fn the_units_cut_the_key_features() {
        let hw = devices::h100();
        for (heads, overhangs) in [
            ((2, 3, 128, 128), false),
            ((2, 3, 64, 64), false),
            ((2, 3, 37, 53), true),
            ((2, 3, 100, 12), true),
            ((2, 3, 12, 8), false),
        ] {
            let problem = walk(1, 7, heads, 1);
            let (blueprint, grid) = CubeDeltaWalkRoutine
                .prepare(&hw, &problem, &CubeDeltaWalkRoutineSettings::DEFAULT)
                .expect("prepares");
            let microkernel = blueprint.microkernel;
            let partitioning = blueprint.partitioning.as_cubek();
            assert_eq!(
                partitioning.leaf().extent(KEY_FEATURE),
                microkernel.shape.rows()
            );
            assert_eq!(partitioning.units() as usize, microkernel.shape.units());
            assert!(microkernel.shape.units().is_power_of_two());
            assert!(
                microkernel.shape.units() * microkernel.shape.rows() >= problem.heads.key_features
            );
            assert_eq!(
                partitioning.overhangs(KEY_FEATURE),
                overhangs,
                "{microkernel} {heads:?}"
            );
            assert_eq!(grid.launch.cube_dim.x as usize, microkernel.shape.units());
        }
    }

    /// A head whose power-of-two units would leave one wholly past its key rows while the rows
    /// divide them is declined by the microkernel's refusal at the floor, however few cells a
    /// unit holds; settings that cut it evenly or raggedly still prepare it.
    #[test]
    fn a_head_whose_units_would_overrun_unmasked_is_declined() {
        let hw = devices::h100();
        let declined = Refusal::Declined {
            why: CubeDeltaWalkMicrokernel::DECLINE,
        };
        for heads in [(2, 3, 12, 8), (2, 3, 40, 36), (2, 3, 100, 12)] {
            let problem = walk(1, 7, heads, 1);
            assert_eq!(
                CubeDeltaWalkRoutine
                    .prepare(&hw, &problem, &CubeDeltaWalkRoutineSettings::FLOOR)
                    .err(),
                Some(declined.clone()),
                "{heads:?}"
            );
            assert_eq!(problem.admit(&hw), Err(declined.clone()), "{heads:?}");
            assert!(
                CubeDeltaWalkRoutine
                    .prepare(&hw, &problem, &CubeDeltaWalkRoutineSettings::DEFAULT)
                    .is_ok(),
                "{heads:?}"
            );
        }
    }

    /// A view starting off a whole line narrows the width the binding settles, so a unit holds
    /// fewer rows to keep its lines within the bound, and more units hold the head.
    #[test]
    fn a_narrower_width_holds_fewer_rows_in_more_units() {
        let hw = devices::h100();
        let settings = CubeDeltaWalkRoutineSettings::DEFAULT;
        let mut problem = walk(1, 7, (2, 3, 128, 128), 1);
        let (aligned, _) = CubeDeltaWalkRoutine
            .prepare(&hw, &problem, &settings)
            .expect("prepares");
        problem.start_bytes.state = 4;
        let (offset, _) = CubeDeltaWalkRoutine
            .prepare(&hw, &problem, &settings)
            .expect("an offset view is served at the same setting");
        assert!(offset.width < aligned.width);
        assert!(offset.microkernel.shape.rows() < aligned.microkernel.shape.rows());
        assert!(offset.microkernel.shape.units() > aligned.microkernel.shape.units());
    }

    /// The shared memory a cube asks is what its microkernel counts at the width the blueprint
    /// settles.
    #[test]
    fn the_cube_asks_the_shared_memory_its_microkernel_counts() {
        let hw = devices::h100();
        for problem in problems() {
            let (blueprint, grid) = CubeDeltaWalkRoutine
                .prepare(&hw, &problem, &CubeDeltaWalkRoutineSettings::DEFAULT)
                .expect("prepares");
            assert_eq!(
                grid.shared_memory,
                blueprint.microkernel.shape.shared_memory(blueprint.runs)
            );
        }
    }

    /// Every setting's estimate, on a prompt and on a decode step: each owner down the walk set its
    /// own part, and the cubes perform at least what the problem asks.
    #[test]
    fn every_routine_estimates_every_setting_whole() {
        let hw = devices::h100();
        let mut prepared = 0;
        for settings in CubeDeltaWalkRoutine.settings(&hw) {
            for problem in problems() {
                let Ok((blueprint, grid)) = CubeDeltaWalkRoutine.prepare(&hw, &problem, &settings)
                else {
                    continue;
                };
                prepared += 1;
                let mut estimate = KernelEstimate::from(&grid);
                CubeDeltaWalkRoutine.estimate(&problem, &blueprint, &grid, &mut estimate);
                assert!(estimate.work.is_some(), "{settings}: no work");
                let staging = estimate.staging.as_ref().expect("staging");
                assert!(
                    staging.stages > 0 && !staging.reads.is_empty(),
                    "{settings}"
                );
                let performed = staging.operations * estimate.cubes.expect("a static grid") as f64;
                let useful = problem.demand().compute_ops as f64;
                assert!(performed >= useful, "{settings}: {performed} < {useful}");
            }
        }
        assert!(prepared > 0);
    }
}
