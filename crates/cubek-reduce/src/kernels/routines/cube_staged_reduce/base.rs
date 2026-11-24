use super::kernel::{cube_staged_reduce_kernel, cube_staged_reduce_with_indices_kernel};
use crate::kernels::{
    BATCH, ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem, estimate_reduce,
    microkernel_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Decline, TileOperand, Vectorization};
use cubecl::tile::{Partitioning, TileTensorLaunch};
use cubek_microkernel::reduce::cube::{
    CubeStagedReduceMicrokernel, CubeStagedReduceMicrokernelSettings,
};
use cubek_microkernel::reduce::{ReduceCarrierFamily, ReduceOperand};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// Every owner a cube, whose units reduce its rows together: few long rows, a cube each. Where its
/// units all read one line of rows, they are the cube's `x`, as many as its microkernel's planes
/// hold, which the hardware splits into planes; where they spread across several, its `x` is a
/// plane. Its cubes run along [`BATCH`].
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
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        let microkernel =
            CubeStagedReduceMicrokernel::best(hw, &problem.work_item(), &settings.microkernel)?;
        let widths = microkernel_widths(hw, problem, &microkernel, microkernel.widest_line())?;
        let microkernel = microkernel.across_rows(hw, widths[0], problem.row_count());
        let partitioning = Partitioning::new(
            problem.space(),
            microkernel
                .levels(&problem.owner_box_at_width(widths[0]))
                .cubes(&[BATCH])
                .build(),
        );
        let cube = CubeResources {
            plane_width: microkernel.launch_plane_width() as u32,
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
        estimate_reduce(problem, &blueprint.microkernel, grid, estimate);
    }
}

impl ReduceLauncher<CubeStagedReduceMicrokernel> for CubeStagedReduceRoutine {
    const NAME: &'static str = "cube-staged-reduce";

    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
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
                    cube_staged_reduce_with_indices_kernel::launch_unchecked::<F, I>(
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
                None => cube_staged_reduce_kernel::launch_unchecked::<F, I>(
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

    /// Four row axes, more than a launch grid has: every row axis is one [`BATCH`](crate::kernels::BATCH),
    /// so it prepares.
    #[test]
    fn prepares_more_row_axes_than_the_grid_holds() {
        let hw = devices::m2_pro();
        let problem = ReduceRoutineProblem::dense(&[2, 2, 2, 2, 64], 4, ReduceOperation::ArgMax);
        let settings = CubeStagedReduceRoutine::settings(&hw)[0];
        let prepared = CubeStagedReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }

    /// tracel:kernel::routine-tested-without-features: no planes, so a Decline.
    #[test]
    fn declines_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = CubeStagedReduceRoutine::settings(&hw)[0];
        let prepared = CubeStagedReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(
            matches!(prepared, Err(Decline::Unserved { .. })),
            "{prepared:?}"
        );
    }

    /// A cube meets only its own planes, so it needs no store reaching another cube.
    #[test]
    fn prepares_without_device_memory_scope() {
        let hw = {
            let mut hw = devices::m2_pro();
            devices::features(&mut hw).device_memory_scope = false;
            hw
        };
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::ArgMax);
        let settings = CubeStagedReduceRoutine::settings(&hw)[0];
        let prepared = CubeStagedReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }
}
