use super::kernel::cube_groups_rms_norm_kernel;
use crate::kernels::{
    RmsNormBinding, RmsNormBlueprint, RmsNormCubeRows, RmsNormRoutine, RmsNormRoutineProblem,
    RmsNormTensors,
};
use core::num::NonZeroUsize;
use cubecl::tile::Levels;
use cubecl::tile::launch::DestinationLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::rms_norm::cube::{
    CubeGroupsRmsNormMicrokernel, CubeGroupsRmsNormMicrokernelSettings,
};
use cubek_microkernel::rms_norm::{R, RmsNormOperand, RmsNormPartitioning};
use cubek_microkernel::row_store::OwnerMemory;
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// The norm with every row owned by a group of units, a cube holding several groups side by
/// side. Into a plain output it asks nothing of the hardware, so it runs everywhere: this kind's
/// floor. Into a packed one it needs plane instructions, and a cube holds the rows its stage fits.
///
/// Each group reads its row where it lies, its units in turns along it, and sums their partials
/// as its microkernel's [`GroupSum`](cubek_microkernel::rms_norm::cube::GroupSum) says: by plane
/// instructions where the device commits to a plane width, else through its own run of the cube's
/// shared memory; a group of one unit owns its row outright.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeGroupsRmsNormRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeGroupsRmsNormRoutineSettings {
    pub microkernel: CubeGroupsRmsNormMicrokernelSettings,
    /// Groups a cube holds, a row each: a target, fewer where the cube's units run out.
    pub groups: NonZeroUsize,
}

impl CubeGroupsRmsNormRoutineSettings {
    /// One group of up to 256 units a cube: a decode step's single row read by as many units as it
    /// has lines, up to a quarter of a cube on every GPU profile.
    pub const DEFAULT: Self = Self {
        microkernel: CubeGroupsRmsNormMicrokernelSettings::DEFAULT,
        groups: NonZeroUsize::MIN,
    };
}

impl CubeGroupsRmsNormRoutineSettings {
    /// Groups a cube of `units`-unit groups holds on `hw`: the target, as many as the cube's units
    /// allow, and no more than `staged`, the owners its store's stage fits.
    fn groups_of(&self, hw: &Hardware, units: usize, staged: usize) -> usize {
        self.groups
            .get()
            .min((hw.properties.hardware.max_units_per_cube as usize / units).max(1))
            .min(staged)
    }
}

impl core::fmt::Display for CubeGroupsRmsNormRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "cube-groups-rms-norm {} groups={}",
            self.microkernel, self.groups
        )
    }
}

pub type CubeGroupsRmsNormRoutineBlueprint = RmsNormBlueprint<CubeGroupsRmsNormMicrokernel>;

impl Routine for CubeGroupsRmsNormRoutine {
    type Problem = RmsNormRoutineProblem;
    type Microkernel = CubeGroupsRmsNormMicrokernel;
    type Settings = CubeGroupsRmsNormRoutineSettings;
    type Blueprint = CubeGroupsRmsNormRoutineBlueprint;
    type Partitioning = RmsNormPartitioning;

    /// The default, then each of its microkernel's smaller groups
    /// ([`CubeGroupsRmsNormMicrokernel::settings`]), more of them a cube, and last one unit owning
    /// one row in a cube of its own: the fewest units a launch spawns, for a row too short to
    /// repay waking more, as a CPU's threads are woken a launch.
    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        use CubeGroupsRmsNormMicrokernelSettings as Group;
        let on = |microkernel, groups| Self::Settings {
            microkernel,
            groups: NonZeroUsize::new(groups).unwrap(),
        };
        vec![
            Self::Settings::DEFAULT,
            on(Group::new(64), 4),
            on(Group::new(16), 16),
            on(Group::new(1), 64),
            on(Group::new(1), 1),
        ]
    }

    /// Nothing of its own: it walks cubes and groups, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &CubeGroupsRmsNormMicrokernel,
        _: &RmsNormPartitioning,
        vectorization: &mut Vectorization<RmsNormOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each cube reads its groups' rows once, straight from global memory, the weight once, and
    /// writes the rows once. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &RmsNormRoutineProblem,
        blueprint: &CubeGroupsRmsNormRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let rows = blueprint.partitioning.cube_box().extent(R);
        estimate.staging = Some(problem.cube_staging(RmsNormCubeRows {
            rows,
            run: problem.dim,
        }));
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }

    /// A group is launched as the cube's `x`, its units the group's, and the groups along `y`. A
    /// group's units are a power of two, so a group within a plane is a run of its lanes and a
    /// larger one is whole planes, which is what its microkernel's plane sums count on.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &RmsNormRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel =
            CubeGroupsRmsNormMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let store = problem.store;
        store.runs_on(hw)?;
        let staged = store.owners_staged(
            hw,
            OwnerMemory {
                staged: problem.dim,
                element: problem.element,
                scratch_bytes: microkernel.shared_memory(),
            },
        );
        let groups = settings.groups_of(hw, microkernel.units, staged);
        let row_box = [vec![(R, 1)], store.width_axes(problem.dim)].concat();
        let partitioning = RmsNormPartitioning::new(
            problem.space(),
            Levels::leaf(&row_box).planes(&[(R, groups)]),
            store,
        );
        let stage = store.stage_bytes(groups * problem.dim, problem.element);
        let cube = CubeResources {
            plane_width: microkernel.units as u32,
            shared_memory: groups * microkernel.shared_memory() + stage,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint =
            RmsNormBlueprint::new(hw, problem, microkernel, partitioning, vectorization)
                .planned_at(microkernel.width)?;
        Ok((blueprint, grid))
    }
}

impl RmsNormRoutine<CubeGroupsRmsNormMicrokernel> for CubeGroupsRmsNormRoutine {
    fn dispatch<O: DestinationLaunch, Out>(
        &self,
        tensors: &RmsNormTensors<Out>,
        binding: RmsNormBinding<O::Operand>,
        blueprint: CubeGroupsRmsNormRoutineBlueprint,
        grid: Grid,
        epsilon: f32,
    ) {
        let problem = tensors.problem();
        let groups = grid.planes().count;
        // SAFETY: every operand is bound through the launcher, and the output through `O`, which
        // mask the reads and writes of a tile the problem's edge cuts short, and the kernel
        // indexes no buffer but through its tiles; its shared memory is the groups' slots,
        // indexed by group and by unit or plane, and a packed store's stage, each group writing its own row,
        // encoded only once the cube has synced.
        unsafe {
            cube_groups_rms_norm_kernel::launch_unchecked::<O>(
                tensors.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.width,
                binding.input.arg(),
                binding.weight.arg(),
                O::arg(binding.out),
                epsilon,
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                groups,
                blueprint.stage,
                problem.element,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_microkernel::rms_norm::HeldRowShape;
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);
    const F32: ElemType = ElemType::Float(FloatKind::F32);

    /// A decode step's model row, a prompt's head rows, a prime row that binds one element at a
    /// time, and a row longer than a cube has units for its lines.
    fn problems() -> [RmsNormRoutineProblem; 4] {
        [
            RmsNormRoutineProblem::dense(1, 2560, F16),
            RmsNormRoutineProblem::dense(512 * 24, 128, F16),
            RmsNormRoutineProblem::dense(2, 111, F32),
            RmsNormRoutineProblem::dense(1, 4100, F32),
        ]
    }

    /// The floor prepares every setting of every problem on every device, plane instructions or
    /// none, a committed width or a range.
    #[test]
    fn the_floor_prepares_everywhere() {
        for hw in [
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
            devices::cdna_wave64(),
            devices::strix_halo(),
            devices::cpu_12(),
        ] {
            for settings in CubeGroupsRmsNormRoutine.settings(&hw) {
                for problem in problems() {
                    let prepared = CubeGroupsRmsNormRoutine.prepare(&hw, &problem, &settings);
                    assert!(prepared.is_ok(), "{settings} on {hw:?}: {prepared:?}");
                }
            }
        }
    }

    /// A group takes no more units than the row has lines at the width it binds at, a power of
    /// two, so a short row's group is small and no unit holds nothing while another holds several;
    /// a prime row binds one element wide, so its lines are its elements.
    #[test]
    fn a_group_is_no_wider_than_its_row() {
        let hw = devices::strix_halo();
        let units = |dim: usize, element| {
            let problem = RmsNormRoutineProblem::dense(1, dim, element);
            let (blueprint, _) = CubeGroupsRmsNormRoutine
                .prepare(&hw, &problem, &CubeGroupsRmsNormRoutineSettings::DEFAULT)
                .expect("the floor prepares");
            blueprint.microkernel.units
        };
        assert_eq!(units(2560, F16), 256);
        assert_eq!(units(128, F16), 16);
        assert_eq!(units(111, F32), 64);
    }

    /// A group too small to hold its row within its registers grows until each unit holds at most
    /// its share's cap, whatever the setting targets: a target is not a demand.
    #[test]
    fn a_group_grows_to_hold_its_row() {
        let hw = devices::strix_halo();
        let one_a_row = CubeGroupsRmsNormRoutineSettings {
            microkernel: CubeGroupsRmsNormMicrokernelSettings {
                units: NonZeroUsize::MIN,
            },
            groups: NonZeroUsize::new(64).unwrap(),
        };
        let units = |dim: usize| {
            let problem = RmsNormRoutineProblem::dense(2, dim, F16);
            let (blueprint, _) = CubeGroupsRmsNormRoutine
                .prepare(&hw, &problem, &one_a_row)
                .expect("the floor prepares");
            blueprint.microkernel.units
        };
        assert_eq!(units(128), 1);
        assert_eq!(units(8192), 64);
        for dim in [128, 2560, 8192] {
            let problem = RmsNormRoutineProblem::dense(2, dim, F16);
            for settings in CubeGroupsRmsNormRoutine.settings(&hw) {
                let (blueprint, _) = CubeGroupsRmsNormRoutine
                    .prepare(&hw, &problem, &settings)
                    .expect("the floor prepares");
                let held = blueprint.microkernel.held_row();
                assert!(
                    held.per_unit() <= HeldRowShape::MAX_LINES_PER_UNIT,
                    "{settings} at {dim}: {held:?}"
                );
            }
        }
    }

    /// A row the width does not divide binds one element at a time, every operand alike.
    #[test]
    fn a_prime_row_binds_one_element_wide() {
        let (blueprint, _) = CubeGroupsRmsNormRoutine
            .prepare(
                &devices::strix_halo(),
                &RmsNormRoutineProblem::dense(2, 111, F32),
                &CubeGroupsRmsNormRoutineSettings::DEFAULT,
            )
            .expect("the floor prepares");
        assert_eq!(blueprint.width, 1);
    }

    /// The default leads the list and no setting is listed twice.
    #[test]
    fn the_default_leads_and_no_setting_repeats() {
        let settings = CubeGroupsRmsNormRoutine.settings(&devices::strix_halo());
        assert_eq!(settings[0], CubeGroupsRmsNormRoutineSettings::DEFAULT);
        let names: std::collections::HashSet<_> =
            settings.iter().map(ToString::to_string).collect();
        assert_eq!(names.len(), settings.len());
    }

    /// Under a packed store a cube holds no more rows than its stage and its groups' own slots
    /// fit in the device's budget, whatever the setting targets; on a device without plane
    /// instructions, which its encoding meets through, it declines.
    #[test]
    fn a_packed_store_takes_the_groups_its_stage_fits() {
        use cubek_microkernel::row_store::{PACKED_STORE_NEEDS_PLANES, RowStore};
        let store = RowStore::Packed {
            block: 32,
            scales_per_word: 4,
        };
        let problem = RmsNormRoutineProblem::dense(512, 4096, F16).stored(store);
        let settings = CubeGroupsRmsNormRoutineSettings {
            microkernel: CubeGroupsRmsNormMicrokernelSettings::new(16),
            groups: NonZeroUsize::new(64).unwrap(),
        };
        let hw = devices::rdna_range();
        let (blueprint, grid) = CubeGroupsRmsNormRoutine
            .prepare(&hw, &problem, &settings)
            .expect("the groups fit a cube's stage");
        let per_group = 4096 * 2 + blueprint.microkernel.shared_memory();
        assert_eq!(grid.planes().count, hw.shared_memory_budget() / per_group);
        assert_eq!(
            CubeGroupsRmsNormRoutine
                .prepare(&devices::wgpu_no_subgroups(), &problem, &settings)
                .err(),
            Some(PACKED_STORE_NEEDS_PLANES)
        );
    }
}
