use super::kernel::{unit_direct_reduce_kernel, unit_direct_reduce_with_indices_kernel};
use crate::kernels::{
    LANE_SLOTS, ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem, reduce_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Refusal, TileOperand, Vectorization};
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceOperand;
use cubek_microkernel::reduce::unit::{UnitReduceMicrokernel, UnitReduceMicrokernelSettings};
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
    pub microkernel: UnitReduceMicrokernelSettings,
    pub planes: usize,
    /// Top-k slots a thread keeps across a line's lanes, which bound a top-k's line width.
    pub lane_slots: usize,
}

impl core::fmt::Display for UnitDirectReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(
            f,
            "{} planes={} lane_slots={}",
            self.microkernel, self.planes, self.lane_slots
        )
    }
}

pub type UnitDirectReduceBlueprint = ReduceBlueprint<UnitReduceMicrokernel>;

impl Routine for UnitDirectReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = UnitReduceMicrokernel;
    type Settings = UnitDirectReduceSettings;
    type Blueprint = UnitDirectReduceBlueprint;
    type Partitioning = Partitioning;

    /// Four planes of units a cube by default, then one and eight.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings> {
        UnitReduceMicrokernel::settings(hw)
            .into_iter()
            .flat_map(|microkernel| [4, 1, 8].map(|planes| (microkernel, planes)))
            .flat_map(|(microkernel, planes)| {
                LANE_SLOTS.map(|lane_slots| UnitDirectReduceSettings {
                    microkernel,
                    planes,
                    lane_slots,
                })
            })
            .collect()
    }

    /// The routine's own code reads nothing: every access is its microkernel's.
    fn vectorize(
        &self,
        microkernel: &UnitReduceMicrokernel,
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
        let microkernel =
            UnitReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        microkernel.vectorize(&(), &mut vectorization);
        let widest = problem
            .operation
            .widest_line_at_lane_slots(settings.lane_slots);
        let widths = reduce_widths(hw, problem, vectorization, widest);
        let plane_width = hw.dispatch_plane_size() as usize;
        let partitioning = Partitioning::new(
            problem.space(),
            Levels::leaf(&problem.owner_box(widths[0]))
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

impl ReduceLauncher<UnitReduceMicrokernel> for UnitDirectReduceRoutine {
    const NAME: &'static str = "unit_direct_reduce";

    fn dispatch<I: TileOperand>(
        &self,
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
        match (binding.indices, problem.indices) {
            (Some(indices), Some(index)) => unsafe {
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
            },
            _ => unsafe {
                unit_direct_reduce_kernel::launch_unchecked::<I>(
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
