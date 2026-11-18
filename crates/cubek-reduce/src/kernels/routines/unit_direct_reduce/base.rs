use super::kernel::{unit_direct_reduce_kernel, unit_direct_reduce_with_indices_kernel};
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem, reduce_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, TileOperand, Vectorization};
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::unit::{
    UnitDirectReduceMicrokernel, UnitDirectReduceMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a unit of the launch, which reduces its rows alone: many short rows, a unit each.
/// The launch groups its units in cubes of `planes` planes as wide as the device dispatches one.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct UnitDirectReduceRoutine;

/// How many planes of units a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceSettings {
    pub microkernel: UnitDirectReduceMicrokernelSettings,
    pub planes: usize,
}

impl core::fmt::Display for UnitDirectReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.planes)
    }
}

pub type UnitDirectReduceBlueprint = ReduceBlueprint<UnitDirectReduceMicrokernel>;

impl Routine for UnitDirectReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = UnitDirectReduceMicrokernel;
    type Settings = UnitDirectReduceSettings;
    type Blueprint = UnitDirectReduceBlueprint;
    type Partitioning = Partitioning;

    /// Four planes of units a cube by default, then one and eight.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        UnitDirectReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| {
                [4, 1, 8].map(|planes| UnitDirectReduceSettings {
                    microkernel,
                    planes,
                })
            })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        microkernel: &UnitDirectReduceMicrokernel,
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
            UnitDirectReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widest = microkernel.widest_line();
        let widths = reduce_widths(hw, problem, vectorization, widest);
        let plane_width = hw.dispatch_plane_size() as usize;
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&problem.owner_box_at_width(widths[0]))
                .launch_units(
                    &problem.owner_dimensions_innermost_first(),
                    UnitCube {
                        plane_width,
                        planes: settings.planes,
                    },
                )
                .build(),
        );
        let cube = CubeResources {
            plane_width: plane_width as u32,
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

impl ReduceLauncher<UnitDirectReduceMicrokernel> for UnitDirectReduceRoutine {
    const NAME: &'static str = "unit-direct-reduce";

    fn dispatch<I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: UnitDirectReduceBlueprint,
        grid: Grid,
    ) {
        let elements = problem.elements;
        // SAFETY: both operands are bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            match binding.indices.zip(problem.indices) {
                Some((indices, index)) => {
                    unit_direct_reduce_with_indices_kernel::launch_unchecked::<I>(
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
                None => unit_direct_reduce_kernel::launch_unchecked::<I>(
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
