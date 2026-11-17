use super::kernel::{cube_reduce_kernel, cube_reduce_with_indices_kernel};
use crate::kernels::{
    ReduceBinding, ReduceRoutine, ReduceRoutineBlueprint, ReduceRoutineProblem, reduce_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, TileOperand, Vectorization};
use cubecl::tile::{Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::cube::{CubeReduceMicrokernel, CubeReduceMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a cube, whose units reduce its rows together: few long rows, a cube each. The
/// cube's units are its `x`, as many as its microkernel's planes hold, and the hardware splits them
/// into its planes. Its cubes run along at most three row dims; it declines more.
#[derive(Debug, Clone, Copy, Default)]
pub struct CubeReduceRoutine;

/// The cube microkernel's planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeReduceSettings {
    pub microkernel: CubeReduceMicrokernelSettings,
}

impl core::fmt::Display for CubeReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.microkernel)
    }
}

pub type CubeReduceBlueprint = ReduceRoutineBlueprint<CubeReduceMicrokernel>;

impl Routine for CubeReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = CubeReduceMicrokernel;
    type Settings = CubeReduceSettings;
    type Blueprint = CubeReduceBlueprint;
    type Partitioning = Partitioning;

    /// Its microkernel's planes: it has no knob of its own.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        CubeReduceMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| CubeReduceSettings { microkernel })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        &self,
        microkernel: &CubeReduceMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<ReduceOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        &self,
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        if problem.owner_axes().len() > MAX_ROW_DIMS {
            return Err(Refusal::Declined {
                why: "the cube reduce runs a cube per box of rows along at most three row dims",
            });
        }
        let microkernel =
            CubeReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widths = reduce_widths(hw, problem, vectorization);
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&problem.owner_box(widths[0]))
                .cubes(&problem.owner_axes())
                .build(),
        );
        let cube = CubeResources {
            plane_width: (microkernel.lanes * microkernel.planes) as u32,
            shared_memory: microkernel.shared_memory_bytes_at_width(widths[0]),
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        Ok((
            ReduceRoutineBlueprint::new(microkernel, partitioning, widths),
            grid,
        ))
    }

    fn estimate(
        &self,
        problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&problem.work_item(), &(), grid.planes(), estimate);
    }
}

/// Most row dims the cubes run along: a launch grid has three.
const MAX_ROW_DIMS: usize = 3;

impl ReduceRoutine<CubeReduceMicrokernel> for CubeReduceRoutine {
    fn dispatch<I: TileOperand>(
        &self,
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: CubeReduceBlueprint,
        grid: Grid,
    ) {
        let elements = problem.elements;
        // SAFETY: both operands are bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; its shared memory is the slots its microkernel declares, which the grid
        // counted.
        match (binding.indices, problem.indices) {
            (Some(indices), Some(index)) => unsafe {
                cube_reduce_with_indices_kernel::launch_unchecked::<I>(
                    client,
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input,
                    binding.out,
                    indices.arg(),
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out, index],
                )
            },
            _ => unsafe {
                cube_reduce_kernel::launch_unchecked::<I>(
                    client,
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input,
                    binding.out,
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out],
                )
            },
        }
    }
}
