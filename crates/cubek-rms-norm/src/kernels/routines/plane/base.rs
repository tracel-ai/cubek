use super::kernel::plane_rms_norm_kernel;
use crate::kernels::{
    RmsNormBinding, RmsNormBlueprint, RmsNormCubeRows, RmsNormRoutine, RmsNormRoutineProblem,
    RmsNormTensors,
};
use core::num::NonZeroUsize;
use cubecl::tile::Levels;
use cubecl::tile::launch::DestinationLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::rms_norm::plane::{
    PlaneRmsNormMicrokernel, PlaneRmsNormMicrokernelSettings,
};
use cubek_microkernel::rms_norm::{D, R, RmsNormOperand, RmsNormPartitioning};
use cubek_microkernel::row_store::{PACKED_STORE_NEEDS_A_CUBE, RowStore};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// The norm with every row owned by a plane of the launch, which groups its planes in cubes of
/// several, consecutive rows a cube. Each plane reads its row once and sums its lanes' partials
/// in one instruction, so no unit waits on another outside its plane, and which cube holds a
/// plane matters to the launch alone.
///
/// It runs where the device commits to a plane width and the row fits the plane's registers.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct PlaneRmsNormRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneRmsNormRoutineSettings {
    pub microkernel: PlaneRmsNormMicrokernelSettings,
    /// Planes a cube holds, a row each.
    pub planes: NonZeroUsize,
}

impl PlaneRmsNormRoutineSettings {
    pub const DEFAULT: Self = Self {
        microkernel: PlaneRmsNormMicrokernelSettings,
        planes: NonZeroUsize::new(4).unwrap(),
    };
}

impl core::fmt::Display for PlaneRmsNormRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "plane-rms-norm {} planes={}",
            self.microkernel, self.planes
        )
    }
}

pub type PlaneRmsNormRoutineBlueprint = RmsNormBlueprint<PlaneRmsNormMicrokernel>;

impl Routine for PlaneRmsNormRoutine {
    type Problem = RmsNormRoutineProblem;
    type Microkernel = PlaneRmsNormMicrokernel;
    type Settings = PlaneRmsNormRoutineSettings;
    type Blueprint = PlaneRmsNormRoutineBlueprint;
    type Partitioning = RmsNormPartitioning;

    /// Planes a cube, the one axis this routine has: one, which spreads a few rows over the most
    /// cubes, to eight, which spawns the fewest for a prompt's many.
    fn settings(&self, _hw: &Hardware) -> Vec<Self::Settings> {
        [4, 1, 2, 8]
            .into_iter()
            .map(|planes| Self::Settings {
                planes: NonZeroUsize::new(planes).unwrap(),
                ..Self::Settings::DEFAULT
            })
            .collect()
    }

    /// Nothing of its own: it walks the launch's planes, and its microkernel makes every access.
    fn vectorize(
        &self,
        microkernel: &PlaneRmsNormMicrokernel,
        _: &RmsNormPartitioning,
        vectorization: &mut Vectorization<RmsNormOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each cube reads its planes' rows once, straight from global memory, the weight once, and
    /// writes the rows once. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &RmsNormRoutineProblem,
        blueprint: &PlaneRmsNormRoutineBlueprint,
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

    /// The plane is launched as wide as the device commits to, its lanes summing across exactly
    /// that many; a row the bound width leaves too many lines a lane for its registers declines.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &RmsNormRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if problem.store != RowStore::Plain {
            return Err(PACKED_STORE_NEEDS_A_CUBE);
        }
        let microkernel =
            PlaneRmsNormMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let partitioning = RmsNormPartitioning::over_planes(
            problem.space(),
            Levels::leaf(&[(R, 1), (D, problem.dim)]),
            PlaneOrder::new(&[(R, settings.planes.get())], CubeOrder::RowMajor),
        );
        let cube = CubeResources {
            plane_width: microkernel.lanes as u32,
            shared_memory: 0,
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

impl RmsNormRoutine<PlaneRmsNormMicrokernel> for PlaneRmsNormRoutine {
    fn dispatch<O: DestinationLaunch, Out>(
        &self,
        tensors: &RmsNormTensors<Out>,
        binding: RmsNormBinding<O::Operand>,
        blueprint: PlaneRmsNormRoutineBlueprint,
        grid: Grid,
        epsilon: f32,
    ) {
        let problem = tensors.problem();
        // SAFETY: every operand is bound through the launcher, and the output through `O`, which
        // mask the reads and writes of a tile the problem's edge cuts short, and the kernel
        // indexes no buffer but through its tiles; it holds no shared memory.
        unsafe {
            plane_rms_norm_kernel::launch_unchecked::<O>(
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
                problem.element,
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

    /// It runs where the device commits to a plane width and declines with its microkernel's
    /// reason where it does not.
    #[test]
    fn it_runs_where_the_device_commits_to_a_width() {
        let problem = RmsNormRoutineProblem::dense(512 * 24, 128, F16);
        for hw in [
            devices::strix_halo(),
            devices::cdna_wave64(),
            devices::h100(),
            devices::wgpu_no_subgroups(),
            devices::rdna_range(),
        ] {
            for settings in PlaneRmsNormRoutine.settings(&hw) {
                match (
                    PlaneRmsNormRoutine.prepare(&hw, &problem, &settings),
                    hw.plane_dim().is_ok(),
                ) {
                    (Ok(_), true) => {}
                    (Err(Refusal::Declined { why: details }), false) => {
                        assert_eq!(details, PlaneRmsNormMicrokernel::DECLINE, "{settings}")
                    }
                    (prepared, _) => panic!("{settings} on {hw:?}: {prepared:?}"),
                }
            }
        }
    }

    /// A row longer than a plane's registers hold is the cube's, and the plane declines it.
    #[test]
    fn a_long_row_outgrows_the_plane() {
        let hw = devices::strix_halo();
        let long = RmsNormRoutineProblem::dense(1, 8192, F16);
        let prepared =
            PlaneRmsNormRoutine.prepare(&hw, &long, &PlaneRmsNormRoutineSettings::DEFAULT);
        assert!(prepared.is_err(), "{prepared:?}");
        let model = RmsNormRoutineProblem::dense(1, 2560, F16);
        assert!(
            PlaneRmsNormRoutine
                .prepare(&hw, &model, &PlaneRmsNormRoutineSettings::DEFAULT)
                .is_ok()
        );
    }
}
