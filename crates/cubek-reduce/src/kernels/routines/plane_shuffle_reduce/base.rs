use super::kernel::{plane_shuffle_reduce_kernel, plane_shuffle_reduce_with_indices_kernel};
use crate::kernels::{
    BATCH, ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem,
    each_microkernel_with, estimate_reduce, microkernel_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Decline, TileOperand, Vectorization};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubecl::tile::{Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::plane::{
    PlaneShuffleReduceMicrokernel, PlaneShuffleReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a plane of the launch, which reduces its rows with the plane microkernel; the
/// launch groups `planes` of them in a cube along the innermost row dimension.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PlaneShuffleReduceRoutine;

/// Its microkernel's settings, and how many of the launch's planes a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleReduceSettings {
    pub microkernel: PlaneShuffleReduceMicrokernelSettings,
    pub planes: usize,
}

impl PlaneShuffleReduceSettings {
    /// Planes a cube holds, each plane its own rows, the likeliest fastest first.
    pub const PLANES: [usize; 3] = [4, 1, 8];
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
        each_microkernel_with(
            PlaneShuffleReduceMicrokernel::settings(hw),
            PlaneShuffleReduceSettings::PLANES,
            |microkernel, planes| PlaneShuffleReduceSettings {
                microkernel,
                planes,
            },
        )
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
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        let microkernel =
            PlaneShuffleReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let widths = microkernel_widths(hw, problem, &microkernel, microkernel.widest_line())?;
        let microkernel = microkernel.across_rows(hw, widths[0], problem.row_count());
        let partitioning = Partitioning::new(
            problem.space(),
            microkernel
                .levels(&problem.owner_box_at_width(widths[0]))
                .launch_planes(
                    &[BATCH],
                    PlaneOrder::new(&[(BATCH, settings.planes)], CubeOrder::RowMajor),
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
        estimate_reduce(problem, &blueprint.microkernel, grid, estimate);
    }
}

impl ReduceLauncher<PlaneShuffleReduceMicrokernel> for PlaneShuffleReduceRoutine {
    const NAME: &'static str = "plane-shuffle-reduce";

    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
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
                    plane_shuffle_reduce_with_indices_kernel::launch_unchecked::<F, I>(
                        client,
                        grid.launch.cube_count,
                        grid.launch.cube_dim,
                        blueprint.width,
                        problem.operation.written_per_row(),
                        binding.input,
                        binding.out,
                        indices.arg(),
                        binding.launcher.partitioning_arg(),
                        blueprint.microkernel,
                        [
                            elements.input,
                            elements.accumulator,
                            elements.out,
                            index.element,
                        ],
                    )
                }
                None => plane_shuffle_reduce_kernel::launch_unchecked::<F, I>(
                    client,
                    grid.launch.cube_count,
                    grid.launch.cube_dim,
                    blueprint.width,
                    problem.operation.written_per_row(),
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::tile::launch::Decline;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// Four row axes, more than a launch grid has: every row axis is one [`BATCH`](crate::kernels::BATCH).
    #[test]
    fn prepares_more_row_axes_than_the_grid_holds() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[2, 2, 2, 2, 64], 4, ReduceOperation::ArgMax);
        let settings = PlaneShuffleReduceRoutine::settings(&hw)[0];
        let prepared = PlaneShuffleReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }

    /// tracel:kernel::routine-tested-without-features: no planes, so a Decline.
    #[test]
    fn declines_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = PlaneShuffleReduceRoutine::settings(&hw)[0];
        let prepared = PlaneShuffleReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(
            matches!(prepared, Err(Decline::Unserved { .. })),
            "{prepared:?}"
        );
    }
}
