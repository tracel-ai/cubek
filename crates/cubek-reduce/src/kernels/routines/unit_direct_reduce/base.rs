use super::kernel::unit_direct_reduce_kernel;
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem, each_microkernel_with,
    estimated_work, unit_launch,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Launcher;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::{Bound, Decline, TileOperand, Vectorization};
use cubek_microkernel::reduce::unit::{
    UnitDirectReduceMicrokernel, UnitDirectReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// Every owner a unit of the launch, which reduces its rows alone: many short rows, a unit each.
/// The launch groups its units in cubes of `plane_count` planes as wide as the device dispatches one.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct UnitDirectReduceRoutine;

/// Its microkernel's settings, and how many planes of units a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceSettings {
    pub microkernel: UnitDirectReduceMicrokernelSettings,
    pub plane_count: usize,
}

impl UnitDirectReduceSettings {
    /// Planes of units a cube holds, each unit its own rows, the likeliest fastest first.
    pub const PLANE_COUNTS: [usize; 3] = [4, 1, 8];
}

impl core::fmt::Display for UnitDirectReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.plane_count)
    }
}

pub type UnitDirectReduceBlueprint = ReduceBlueprint<UnitDirectReduceMicrokernel>;

impl Routine for UnitDirectReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = UnitDirectReduceMicrokernel;
    type Settings = UnitDirectReduceSettings;
    type Blueprint = UnitDirectReduceBlueprint;
    type Partitioning = ();

    /// Four planes of units a cube by default, then one and eight.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        each_microkernel_with(
            UnitDirectReduceMicrokernel::settings(hw),
            UnitDirectReduceSettings::PLANE_COUNTS,
            |microkernel, plane_count| UnitDirectReduceSettings {
                microkernel,
                plane_count,
            },
        )
    }

    fn vectorize(
        microkernel: &UnitDirectReduceMicrokernel,
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
        let (microkernel, lane_count) =
            best_microkernel!(UnitDirectReduceMicrokernel, hw, problem, settings);
        let (partitioning, grid) = unit_launch(hw, problem, lane_count, settings.plane_count)?;
        Ok((
            ReduceBlueprint {
                microkernel,
                partitioning,
                lane_count,
            },
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

impl ReduceRoutine<UnitDirectReduceMicrokernel> for UnitDirectReduceRoutine {
    const NAME: &'static str = "unit-direct-reduce";

    type Out = TileTensorLaunch;

    fn out_arg(out: Bound) -> TileTensorLaunch {
        out.arg()
    }

    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        launcher: &Launcher,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: UnitDirectReduceBlueprint,
        grid: Grid,
    ) -> Result<(), Decline> {
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; it holds no shared memory.
        unsafe {
            unit_direct_reduce_kernel::launch_unchecked::<F, I>(
                client,
                grid.launch.cube_count,
                grid.launch.cube_dim,
                blueprint.lane_count,
                problem.operation.written_per_row(),
                binding.input,
                binding.out,
                binding.indices.map(Bound::arg).into(),
                launcher.partitioning_arg(),
                blueprint.microkernel,
                problem.compiled_elements(),
            )
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::launch_first_served;
    use crate::kernels::testing::{BufferLaunch, UnboundLaunch, launch_unbound, swept_problems};
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    prepares_more_row_axes_than_the_grid_holds!(UnitDirectReduceRoutine);
    prepares_lacking_every_feature!(UnitDirectReduceRoutine);
    names_its_setting!(UnitDirectReduceRoutine, "lane_slots=32 planes=4");
    estimates_its_microkernel!(UnitDirectReduceRoutine, ReduceRoutineProblem::work_item);

    /// Checks that the routine, which runs where every other declines, prepares every problem on
    /// `hw`.
    fn prepares_every_problem(hw: &Hardware) {
        for problem in swept_problems() {
            let prepared = UnitDirectReduceRoutine::prepare_first_served(hw, &problem);
            assert!(prepared.is_ok(), "{problem:?}");
        }
    }

    on_each_device!(the_unit_routine, prepares_every_problem);

    /// Checks that an argmax of `shape` along `axis` compiles on an M2 Pro the microkernel, lanes
    /// and kernel of `other`, a shape of its rank and layout: the rows and the row length are the
    /// launch's.
    fn compiles_what_another_shape_compiles(shape: &[usize], other: &[usize], axis: usize) {
        const LANE_COUNT: usize = 4;
        let hw = devices::m2_pro();
        let [problem, other] = [shape, other]
            .map(|shape| ReduceRoutineProblem::dense(shape, axis, ReduceOperation::ArgMax));
        let settings = UnitDirectReduceRoutine::settings(&hw)[0];
        let microkernel =
            UnitDirectReduceMicrokernel::best(&hw, &other.work_item(), &settings.microkernel)
                .expect("the unit microkernel runs everywhere");
        let (partitioning, _) = unit_launch(&hw, &other, LANE_COUNT, settings.plane_count)
            .expect("the unit launch fits an M2 Pro");
        let (blueprint, _) = UnitDirectReduceRoutine::prepare(&hw, &problem, &settings)
            .expect("the unit routine runs everywhere");
        assert_eq!(
            (
                blueprint.microkernel,
                blueprint.lane_count,
                problem.kernel(&blueprint.partitioning)
            ),
            (microkernel, LANE_COUNT, other.kernel(&partitioning))
        );
    }

    /// A sequence that grows by a token recompiles nothing.
    #[test]
    fn rows_along_the_contiguous_dim_compile_one_kernel_for_every_shape() {
        compiles_what_another_shape_compiles(&[2, 5, 300], &[3, 7, 308], 2);
    }

    #[test]
    fn rows_across_the_contiguous_dim_compile_one_kernel_for_every_shape() {
        compiles_what_another_shape_compiles(&[2, 5, 300], &[4, 9, 308], 1);
    }

    const LAUNCH: BufferLaunch = launch_first_served::<UnitDirectReduceRoutine, _>;
    const UNBOUND: UnboundLaunch = launch_unbound::<UnitDirectReduceRoutine, _>;

    reduces_each_dim!(LAUNCH, UNBOUND);
    reduces_more_row_axes_than_the_grid_has!(LAUNCH);
    reduces_with_indices!(LAUNCH);
}
