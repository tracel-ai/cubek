use super::kernel::cube_rms_norm_kernel;
use crate::kernels::{
    RmsNormBinding, RmsNormBlueprint, RmsNormCubeRows, RmsNormRoutine, RmsNormRoutineProblem,
    RmsNormTensors,
};
use cubecl::tile::Levels;
use cubecl::tile::launch::DestinationLaunch;
use cubecl::tile::launch::Refusal;
use cubecl::tile::launch::Vectorization;
use cubek_microkernel::rms_norm::cube::{CubeRmsNormMicrokernel, CubeRmsNormMicrokernelSettings};
use cubek_microkernel::rms_norm::{R, RmsNormOperand, RmsNormPartitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// The norm with every row owned by a cube, its planes each summing a contiguous run of the row
/// and meeting once in shared memory. What a single row of the model's width wants: more units
/// on it than a plane has, and a meeting of a few planes rather than of every unit.
///
/// It runs where its plane microkernel does.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct CubeRmsNormRoutine;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeRmsNormRoutineSettings {
    pub microkernel: CubeRmsNormMicrokernelSettings,
}

impl CubeRmsNormRoutineSettings {
    pub const DEFAULT: Self = Self {
        microkernel: CubeRmsNormMicrokernelSettings::DEFAULT,
    };
}

impl core::fmt::Display for CubeRmsNormRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "cube-rms-norm {}", self.microkernel)
    }
}

pub type CubeRmsNormRoutineBlueprint = RmsNormBlueprint<CubeRmsNormMicrokernel>;

impl Routine for CubeRmsNormRoutine {
    type Problem = RmsNormRoutineProblem;
    type Microkernel = CubeRmsNormMicrokernel;
    type Settings = CubeRmsNormRoutineSettings;
    type Blueprint = CubeRmsNormRoutineBlueprint;
    type Partitioning = RmsNormPartitioning;

    /// Its microkernel's settings ([`CubeRmsNormMicrokernel::settings`]): it has no knob of its own.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeRmsNormMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| Self::Settings { microkernel })
            .collect()
    }

    /// Nothing of its own: it walks cubes, and its microkernel walks their planes and makes every
    /// access.
    fn vectorize(
        &self,
        microkernel: &CubeRmsNormMicrokernel,
        _: &RmsNormPartitioning,
        vectorization: &mut Vectorization<RmsNormOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Each cube reads its row once, a plane a run, straight from global memory, the weight once,
    /// and writes the row once. Then its microkernel, for the work.
    fn estimate(
        &self,
        problem: &RmsNormRoutineProblem,
        blueprint: &CubeRmsNormRoutineBlueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        estimate.staging = Some(problem.cube_staging(RmsNormCubeRows {
            rows: 1,
            run: blueprint.microkernel.plane.run,
        }));
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }

    /// The planes are launched as wide as the device commits to, along `y`, one run of the row
    /// each; a run the bound width leaves too many lines a lane for its registers declines.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &RmsNormRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel =
            CubeRmsNormMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let store = problem.store;
        store.runs_on(hw)?;
        let plane_box = [vec![(R, 1)], store.width_axes(microkernel.plane.run)].concat();
        let partitioning = RmsNormPartitioning::new(
            problem.space(),
            Levels::leaf(&plane_box).planes(&[(store.cut(), microkernel.planes)]),
            store,
        );
        let stage = store.stage_bytes(problem.dim, problem.element);
        let cube = CubeResources {
            plane_width: microkernel.plane.lanes as u32,
            shared_memory: microkernel.shared_memory() + stage,
        };
        let grid = Grid::new(hw, partitioning.as_tile(), cube)?;
        let mut vectorization = Vectorization::new();
        self.vectorize(&microkernel, &partitioning, &mut vectorization);
        let blueprint =
            RmsNormBlueprint::new(hw, problem, microkernel, partitioning, vectorization)
                .planned_at(microkernel.plane.width)?;
        Ok((blueprint, grid))
    }
}

impl RmsNormRoutine<CubeRmsNormMicrokernel> for CubeRmsNormRoutine {
    fn dispatch<O: DestinationLaunch, Out>(
        &self,
        tensors: &RmsNormTensors<Out>,
        binding: RmsNormBinding<O::Operand>,
        blueprint: CubeRmsNormRoutineBlueprint,
        grid: Grid,
        epsilon: f32,
    ) {
        let problem = tensors.problem();
        // SAFETY: every operand is bound through the launcher, and the output through `O`, which
        // mask the reads and writes of a tile the problem's edge cuts short, and the kernel
        // indexes no buffer but through its tiles; its shared memory is a slot a plane, indexed
        // by plane, and a packed store's stage, each plane writing its own run, encoded only once
        // the cube has synced.
        unsafe {
            cube_rms_norm_kernel::launch_unchecked::<O>(
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
                blueprint.stage,
                problem.element,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use core::num::NonZeroUsize;
    use cubecl::ir::{ElemType, FloatKind};
    use cubek_std::devices;

    const F16: ElemType = ElemType::Float(FloatKind::F16);

    /// A row of the model's width is split into runs that divide it and leave every lane a line,
    /// whatever the target.
    #[test]
    fn the_planes_divide_the_row() {
        let hw = devices::strix_halo();
        let planes = |dim: usize, target: usize| {
            let settings = CubeRmsNormRoutineSettings {
                microkernel: CubeRmsNormMicrokernelSettings {
                    planes: NonZeroUsize::new(target).unwrap(),
                },
            };
            let (blueprint, _) = CubeRmsNormRoutine
                .prepare(&hw, &RmsNormRoutineProblem::dense(1, dim, F16), &settings)
                .expect("a committed wave32 runs the cube");
            let microkernel = blueprint.microkernel;
            (microkernel.planes, microkernel.plane.run)
        };
        assert_eq!(planes(2560, 4), (4, 640));
        assert_eq!(planes(2560, 16), (10, 256));
        assert_eq!(planes(1024, 16), (4, 256));
        assert_eq!(planes(128, 4), (1, 128));
    }

    /// It declines where its plane microkernel does: a device committing to no width.
    #[test]
    fn it_runs_where_its_plane_does() {
        let problem = RmsNormRoutineProblem::dense(1, 2560, F16);
        for hw in [devices::wgpu_no_subgroups(), devices::rdna_range()] {
            let prepared =
                CubeRmsNormRoutine.prepare(&hw, &problem, &CubeRmsNormRoutineSettings::DEFAULT);
            assert!(
                matches!(prepared, Err(Refusal::Declined { why: details }) if details == CubeRmsNormMicrokernel::DECLINE),
                "{hw:?}"
            );
        }
        for hw in [
            devices::strix_halo(),
            devices::cdna_wave64(),
            devices::h100(),
        ] {
            assert!(
                CubeRmsNormRoutine
                    .prepare(&hw, &problem, &CubeRmsNormRoutineSettings::DEFAULT)
                    .is_ok()
            );
        }
    }
}
