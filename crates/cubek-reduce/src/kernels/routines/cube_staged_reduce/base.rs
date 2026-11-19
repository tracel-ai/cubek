use super::kernel::{cube_staged_reduce_kernel, cube_staged_reduce_with_indices_kernel};
use crate::kernels::{
    BATCH, ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem, reduce_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, TileOperand, Vectorization};
use cubecl::tile::{Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::cube::{
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a cube, whose units reduce its rows together: few long rows, a cube each. The
/// cube's units are its `x`, as many as its microkernel's planes hold, and the hardware splits them
/// into its planes. Its cubes run along at most three row dims; it declines more.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct CubeStagedReduceRoutine;

/// The cube microkernel's planes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeStagedReduceSettings {
    pub microkernel: CubeStagedReduceMicrokernelSettings,
}

impl core::fmt::Display for CubeStagedReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.microkernel)
    }
}

pub type CubeStagedReduceBlueprint = ReduceBlueprint<CubeStagedReduceMicrokernel>;

impl Routine for CubeStagedReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = CubeStagedReduceMicrokernel;
    type Settings = CubeStagedReduceSettings;
    type Blueprint = CubeStagedReduceBlueprint;
    type Partitioning = Partitioning;

    /// Its microkernel's planes: it has no knob of its own.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| CubeStagedReduceSettings { microkernel })
            .collect()
    }

    fn vectorize(
        microkernel: &CubeStagedReduceMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<ReduceOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    fn prepare(
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let microkernel =
            CubeStagedReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widest = microkernel.widest_line();
        let widths = reduce_widths(hw, problem, vectorization, widest);
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&problem.owner_box_at_width(widths[0]))
                .cubes(&[BATCH])
                .build(),
        );
        let cube = CubeResources {
            plane_width: (microkernel.lanes() * microkernel.planes()) as u32,
            shared_memory: microkernel.shared_memory_bytes_at_width(widths[0]),
        };
        let grid = Grid::new(hw, &partitioning, cube)?;
        Ok((
            ReduceBlueprint::new(microkernel, partitioning, widths),
            grid,
        ))
    }

    fn estimate(
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

impl ReduceLauncher<CubeStagedReduceMicrokernel> for CubeStagedReduceRoutine {
    const NAME: &'static str = "cube-staged-reduce";

    fn dispatch<I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: CubeStagedReduceBlueprint,
        grid: Grid,
    ) {
        let elements = problem.elements;
        // SAFETY: both operands are bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; its shared memory is the slots its microkernel declares, which the grid
        // counted.
        unsafe {
            match binding.indices.zip(problem.indices) {
                Some((indices, index)) => {
                    cube_staged_reduce_with_indices_kernel::launch_unchecked::<I>(
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
                }
                None => cube_staged_reduce_kernel::launch_unchecked::<I>(
                    client,
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    binding.input,
                    binding.out,
                    binding.launcher.partitioning_arg(),
                    blueprint.microkernel,
                    [elements.input, elements.accumulator, elements.out],
                ),
            }
        }
    }
}
