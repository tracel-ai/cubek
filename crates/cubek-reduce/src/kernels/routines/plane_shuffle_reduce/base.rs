use super::kernel::{plane_shuffle_reduce_kernel, plane_shuffle_reduce_with_indices_kernel};
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem, reduce_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, TileOperand, Vectorization};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::plane::{
    PlaneShuffleReduceMicrokernel, PlaneShuffleReduceMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a plane of the launch, which reduces its rows with the plane microkernel; the
/// launch groups `planes` of them in a cube along the innermost row dimension.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PlaneShuffleReduceRoutine;

/// How many of the launch's planes a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleReduceSettings {
    pub microkernel: PlaneShuffleReduceMicrokernelSettings,
    pub planes: usize,
}

impl core::fmt::Display for PlaneShuffleReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.planes)
    }
}

pub type PlaneShuffleReduceBlueprint = ReduceBlueprint<PlaneShuffleReduceMicrokernel>;

impl Routine for PlaneShuffleReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = PlaneShuffleReduceMicrokernel;
    type Settings = PlaneShuffleReduceSettings;
    type Blueprint = PlaneShuffleReduceBlueprint;
    type Partitioning = Partitioning;

    /// Four planes a cube by default, then one and eight.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        PlaneShuffleReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                [4, 1, 8].map(|planes| PlaneShuffleReduceSettings {
                    microkernel,
                    planes,
                })
            })
            .collect()
    }

    fn vectorize(
        microkernel: &PlaneShuffleReduceMicrokernel,
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
            PlaneShuffleReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widest = microkernel.widest_line();
        let widths = reduce_widths(hw, problem, vectorization, widest);
        let owners = problem.owner_dimensions_innermost_first();
        let innermost = *owners
            .first()
            .expect("a problem places its owners along a dimension");
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&problem.owner_box_at_width(widths[0]))
                .launch_planes(
                    &owners,
                    PlaneOrder::new(&[(innermost, settings.planes)], CubeOrder::RowMajor),
                )
                .build(),
        );
        let cube = CubeResources {
            plane_width: microkernel.lanes() as u32,
            shared_memory: 0,
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

impl ReduceLauncher<PlaneShuffleReduceMicrokernel> for PlaneShuffleReduceRoutine {
    const NAME: &'static str = "plane-shuffle-reduce";

    fn dispatch<I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: PlaneShuffleReduceBlueprint,
        grid: Grid,
    ) {
        let elements = problem.elements;
        // SAFETY: both operands are bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            match binding.indices.zip(problem.indices) {
                Some((indices, index)) => {
                    plane_shuffle_reduce_with_indices_kernel::launch_unchecked::<I>(
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
                None => plane_shuffle_reduce_kernel::launch_unchecked::<I>(
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
