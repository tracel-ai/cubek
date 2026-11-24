use super::kernel::plane_shuffle_reduce_kernel;
use crate::kernels::{
    BATCH, ReduceBinding, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem,
    each_microkernel_with, estimated_work, placed,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::{Bound, Decline, TileOperand, Vectorization};
use cubecl::tile::space::{CubeOrder, PlaneOrder};
use cubek_microkernel::reduce::plane::{
    PlaneShuffleReduceMicrokernel, PlaneShuffleReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a plane of the launch, which reduces its rows with the plane microkernel; the
/// launch groups `plane_count` of them in a cube along the innermost row dimension.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct PlaneShuffleReduceRoutine;

/// Its microkernel's settings, and how many of the launch's planes a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneShuffleReduceSettings {
    pub microkernel: PlaneShuffleReduceMicrokernelSettings,
    pub plane_count: usize,
}

impl PlaneShuffleReduceSettings {
    /// Planes a cube holds, each plane its own rows, the likeliest fastest first.
    pub const PLANE_COUNTS: [usize; 3] = [4, 1, 8];
}

impl core::fmt::Display for PlaneShuffleReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.plane_count)
    }
}

pub type PlaneShuffleReduceBlueprint = ReduceBlueprint<PlaneShuffleReduceMicrokernel>;

impl Routine for PlaneShuffleReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = PlaneShuffleReduceMicrokernel;
    type Settings = PlaneShuffleReduceSettings;
    type Blueprint = PlaneShuffleReduceBlueprint;
    type Partitioning = ();

    /// Four planes a cube by default, then one and eight.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        each_microkernel_with(
            PlaneShuffleReduceMicrokernel::settings(hw),
            PlaneShuffleReduceSettings::PLANE_COUNTS,
            |microkernel, plane_count| PlaneShuffleReduceSettings {
                microkernel,
                plane_count,
            },
        )
    }

    fn vectorize(
        microkernel: &PlaneShuffleReduceMicrokernel,
        _partitioning: &(),
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
        let lane_counts = Self::lane_counts(hw, problem, &microkernel, microkernel.widest_line())?;
        let microkernel = microkernel.across_rows(hw, lane_counts[0], problem.row_count());
        let levels = microkernel
            .levels(&problem.owner_box_at_lane_count(lane_counts[0]))
            .launch_planes(
                &[BATCH],
                PlaneOrder::new(&[(BATCH, settings.plane_count)], CubeOrder::RowMajor),
            )
            .build();
        let cube = CubeResources {
            plane_width: microkernel.lanes() as u32,
            shared_memory: 0,
        };
        let (partitioning, grid) = placed(hw, problem.space(), levels, cube)?;
        Ok((
            ReduceBlueprint::new(microkernel, partitioning, lane_counts),
            grid,
        ))
    }

    fn estimate(
        problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        let (item, planes) = estimated_work(problem, grid);
        blueprint.microkernel.estimate(&item, &(), planes, estimate);
    }
}

impl ReduceRoutine<PlaneShuffleReduceMicrokernel> for PlaneShuffleReduceRoutine {
    const NAME: &'static str = "plane-shuffle-reduce";

    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: PlaneShuffleReduceBlueprint,
        grid: Grid,
    ) {
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            plane_shuffle_reduce_kernel::launch_unchecked::<F, I>(
                client,
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.lane_count,
                problem.operation.written_per_row(),
                binding.input,
                binding.out,
                binding.indices.map(Bound::arg).into(),
                binding.launcher.partitioning_arg(),
                blueprint.microkernel,
                problem.compiled_elements(),
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::launch_first_served;
    use crate::kernels::testing::{BufferLaunch, UnboundLaunch, launch_unbound};
    use cubecl::tile::launch::Decline;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    #[test]
    fn prepares_more_row_axes_than_the_grid_holds() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[2, 2, 2, 2, 64], 4, ReduceOperation::ArgMax);
        let settings = PlaneShuffleReduceRoutine::settings(&hw)[0];
        let prepared = PlaneShuffleReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }

    /// A plane needs the plane instructions its lanes meet by.
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

    const LAUNCH: BufferLaunch = launch_first_served::<PlaneShuffleReduceRoutine, _>;
    const UNBOUND: UnboundLaunch = launch_unbound::<PlaneShuffleReduceRoutine, _>;

    reduces_each_dim!(LAUNCH, UNBOUND);
    reduces_more_row_axes_than_the_grid_has!(LAUNCH);
}
