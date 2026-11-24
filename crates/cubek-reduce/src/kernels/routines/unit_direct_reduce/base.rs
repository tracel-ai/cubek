use super::kernel::{unit_direct_reduce_kernel, unit_direct_reduce_with_indices_kernel};
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem, each_microkernel_with,
    estimate_reduce, microkernel_widths, unit_launch,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Decline, TileOperand, Vectorization};
use cubecl::tile::{Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::unit::{
    UnitDirectReduceMicrokernel, UnitDirectReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// Every owner a unit of the launch, which reduces its rows alone: many short rows, a unit each.
/// The launch groups its units in cubes of `planes` planes as wide as the device dispatches one.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct UnitDirectReduceRoutine;

/// Its microkernel's settings, and how many planes of units a cube holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectReduceSettings {
    pub microkernel: UnitDirectReduceMicrokernelSettings,
    pub planes: usize,
}

impl UnitDirectReduceSettings {
    /// Planes of units a cube holds, each unit its own rows, the likeliest fastest first.
    pub const PLANES: [usize; 3] = [4, 1, 8];
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
        each_microkernel_with(
            UnitDirectReduceMicrokernel::settings(hw),
            UnitDirectReduceSettings::PLANES,
            |microkernel, planes| UnitDirectReduceSettings {
                microkernel,
                planes,
            },
        )
    }

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
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        let microkernel =
            UnitDirectReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let widths = microkernel_widths(hw, problem, &microkernel, microkernel.widest_line())?;
        let (partitioning, grid) = unit_launch(hw, problem, widths[0], settings.planes)?;
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

impl ReduceRoutine<UnitDirectReduceMicrokernel> for UnitDirectReduceRoutine {
    const NAME: &'static str = "unit-direct-reduce";

    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
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
                    unit_direct_reduce_with_indices_kernel::launch_unchecked::<F, I>(
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
                None => unit_direct_reduce_kernel::launch_unchecked::<F, I>(
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
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// Checks that an argmax of `shape` along `axis` compiles on an M2 Pro the microkernel, width
    /// and kernel of `other`, a shape of its rank and layout: the rows and the row length are the
    /// launch's.
    fn compiles_what_another_shape_compiles(shape: &[usize], other: &[usize], axis: usize) {
        const WIDTH: usize = 4;
        let hw = devices::m2_pro();
        let [problem, other] = [shape, other]
            .map(|shape| ReduceRoutineProblem::dense(shape, axis, ReduceOperation::ArgMax));
        let settings = UnitDirectReduceRoutine::settings(&hw)[0];
        let microkernel =
            UnitDirectReduceMicrokernel::best(&hw, &other.work_item(), &settings.microkernel)
                .expect("the unit microkernel runs everywhere");
        let (partitioning, _) = unit_launch(&hw, &other, WIDTH, settings.planes)
            .expect("the unit launch fits an M2 Pro");
        let (blueprint, _) = UnitDirectReduceRoutine::prepare(&hw, &problem, &settings)
            .expect("the unit routine runs everywhere");
        assert_eq!(
            (
                blueprint.microkernel,
                blueprint.width,
                problem.kernel(&blueprint.partitioning)
            ),
            (microkernel, WIDTH, other.kernel(&partitioning))
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

    /// Four row axes, more than a launch grid has: every row axis is one [`BATCH`](crate::kernels::BATCH).
    #[test]
    fn prepares_more_row_axes_than_the_grid_holds() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[2, 2, 2, 2, 64], 4, ReduceOperation::ArgMax);
        let settings = UnitDirectReduceRoutine::settings(&hw)[0];
        let prepared = UnitDirectReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }

    /// The unit routine reads no Feature.
    #[test]
    fn prepares_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = UnitDirectReduceRoutine::settings(&hw)[0];
        let prepared = UnitDirectReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }
}
