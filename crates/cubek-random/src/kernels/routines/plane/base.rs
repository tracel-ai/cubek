use super::kernel::plane_random_kernel;
use crate::kernels::{RandomRoutine, RandomRoutineProblem, RandomTensor, VALUES};
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::random::plane::{PlaneRandomMicrokernel, PlaneRandomMicrokernelSettings};
use cubek_microkernel::random::{Distribution, SeedsLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every plane of the launch a run of the output, its lanes interleaving their draws so the plane
/// writes consecutive lines at every draw: what a GPU's memory coalesces.
#[derive(Debug, Clone, Copy, Default)]
pub struct PlaneRandomRoutine;

/// How many planes a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneRandomRoutineSettings {
    pub microkernel: PlaneRandomMicrokernelSettings,
    pub planes: usize,
}

impl core::fmt::Display for PlaneRandomRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.planes)
    }
}

/// What a launch of [`PlaneRandomRoutine`] compiles.
#[derive(Debug, Clone, PartialEq)]
pub struct PlaneRandomRoutineBlueprint {
    pub microkernel: PlaneRandomMicrokernel,
    pub partitioning: Partitioning,
    /// The width the output binds at.
    pub width: usize,
}

impl Routine for PlaneRandomRoutine {
    type Problem = RandomRoutineProblem;
    type Microkernel = PlaneRandomMicrokernel;
    type Settings = PlaneRandomRoutineSettings;
    type Blueprint = PlaneRandomRoutineBlueprint;
    type Partitioning = Partitioning;

    /// Eight planes a cube by default, then four and sixteen.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        PlaneRandomMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                [8, 4, 16].map(|planes| PlaneRandomRoutineSettings {
                    microkernel,
                    planes,
                })
            })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        &self,
        microkernel: &PlaneRandomMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<()>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &RandomRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel = PlaneRandomMicrokernel::best(hw, &(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let width = problem.width(hw.vector_ceiling(problem.element), vectorization);
        let lanes = hw.dispatch_plane_size() as usize;
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&[(VALUES, lanes * microkernel.values_per_lane)])
                .launch_planes(
                    &[VALUES],
                    PlaneOrder::new(&[(VALUES, settings.planes)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let cube = CubeResources {
            plane_width: lanes as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        let blueprint = PlaneRandomRoutineBlueprint {
            microkernel,
            partitioning,
            width,
        };
        Ok((blueprint, grid))
    }

    fn estimate(
        &self,
        _problem: &RandomRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&(), &(), grid.planes(), estimate);
    }
}

impl RandomRoutine for PlaneRandomRoutine {
    fn dispatch<D: Distribution>(
        &self,
        tensor: &RandomTensor,
        args: <D::Args as LaunchArg>::RuntimeArg,
        seeds: SeedsLaunch,
        blueprint: PlaneRandomRoutineBlueprint,
        grid: Grid,
    ) {
        let binding = tensor.bind(&blueprint.partitioning, blueprint.width, &grid);
        // SAFETY: the output is bound through the launcher, which masks the writes of a run the
        // output's end cuts short, and the kernel indexes no buffer but through its tile; it
        // holds no shared memory.
        unsafe {
            plane_random_kernel::launch_unchecked::<D>(
                tensor.client(),
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.width,
                binding.output.arg(),
                binding.launcher.partitioning_arg(),
                args,
                seeds,
                blueprint.microkernel,
                tensor.problem().element,
            );
        }
    }
}
