use super::kernel::unit_random_kernel;
use crate::kernels::{RandomRoutine, RandomRoutineProblem, RandomTensor, VALUES};
use cubecl::ir::{ElemType, UIntKind};
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, Vectorization};
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Levels, Partitioning};
use cubek_microkernel::random::unit::{UnitRandomMicrokernel, UnitRandomMicrokernelSettings};
use cubek_microkernel::random::{Distribution, SeedsLaunch};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every unit of the launch a run of the output of its own, drawn line after line: on a CPU, where
/// a unit is a core, the line is the only SIMD there is and two cores must never write the same
/// cache line.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnitRandomRoutine;

/// A unit's run is read off the device, so the routine has nothing of its own a measurement
/// settles.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitRandomRoutineSettings {
    pub microkernel: UnitRandomMicrokernelSettings,
}

impl core::fmt::Display for UnitRandomRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.microkernel.fmt(f)
    }
}

/// What a launch of [`UnitRandomRoutine`] compiles.
#[derive(Debug, Clone, PartialEq)]
pub struct UnitRandomRoutineBlueprint {
    pub microkernel: UnitRandomMicrokernel,
    pub partitioning: Partitioning,
    /// The width the output binds at.
    pub width: usize,
}

impl Routine for UnitRandomRoutine {
    type Problem = RandomRoutineProblem;
    type Microkernel = UnitRandomMicrokernel;
    type Settings = UnitRandomRoutineSettings;
    type Blueprint = UnitRandomRoutineBlueprint;
    type Partitioning = Partitioning;

    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        UnitRandomMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| UnitRandomRoutineSettings { microkernel })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        &self,
        microkernel: &UnitRandomMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<()>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// A unit's run is the smallest worth a threadpool wake-up: a dispatch costs about a
    /// microsecond, which at tens of GB/s of stores is tens of KB, and half the L1d, which the CPU
    /// runtime reports as its shared memory, lands there. Its line is the one a register holds the
    /// generator at: four `u32` registers advance per draw against a single store.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &RandomRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel = UnitRandomMicrokernel::best(hw, &(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let width = problem.width(
            hw.vector_ceiling(ElemType::UInt(UIntKind::U32)),
            vectorization,
        );
        let run_bytes = hw.properties.hardware.max_shared_memory_size / 2;
        let draws = run_bytes
            .div_ceil(width * problem.element.size() * problem.lines_per_draw)
            .max(1);
        let plane_width = hw.dispatch_plane_size() as usize;
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&[(VALUES, draws * problem.lines_per_draw * width)])
                .launch_units(
                    &[VALUES],
                    UnitCube {
                        plane_width,
                        planes: hw.full_cube_planes(),
                    },
                )
                .build(),
        );
        let cube = CubeResources {
            plane_width: plane_width as u32,
            shared_memory: 0,
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        let blueprint = UnitRandomRoutineBlueprint {
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

impl RandomRoutine for UnitRandomRoutine {
    fn dispatch<D: Distribution>(
        &self,
        tensor: &RandomTensor,
        args: <D::Args as LaunchArg>::RuntimeArg,
        seeds: SeedsLaunch,
        blueprint: UnitRandomRoutineBlueprint,
        grid: Grid,
    ) {
        let binding = tensor.bind(&blueprint.partitioning, blueprint.width, &grid);
        // SAFETY: the output is bound through the launcher, which masks the writes of a run the
        // output's end cuts short, and the kernel indexes no buffer but through its tile; it
        // holds no shared memory.
        unsafe {
            unit_random_kernel::launch_unchecked::<D>(
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
