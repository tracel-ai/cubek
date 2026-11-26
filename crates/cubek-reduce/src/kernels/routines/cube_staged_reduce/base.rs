use super::kernel::cube_staged_reduce_kernel;
use crate::kernels::{
    BATCH, ReduceBinding, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem, estimated_work,
    placed,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Launcher;
use cubecl::tile::TileTensorLaunch;
use cubecl::tile::launch::{Bound, Decline, TileOperand, Vectorization};
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
    type Partitioning = ();

    /// Its microkernel's planes: it has no knob of its own.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        CubeStagedReduceMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| CubeStagedReduceSettings { microkernel })
            .collect()
    }

    fn vectorize(
        microkernel: &CubeStagedReduceMicrokernel,
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
            best_microkernel!(CubeStagedReduceMicrokernel, hw, problem, settings);
        let microkernel = microkernel.across_rows(hw, lane_count, problem.row_count());
        let levels = microkernel
            .levels(&problem.owner_box_at_lane_count(lane_count))
            .cubes(&[BATCH])
            .build();
        let cube = CubeResources {
            plane_width: microkernel.launch_plane_width() as u32,
            shared_memory: microkernel.shared_memory_bytes_at_lane_count(lane_count),
        };
        let (partitioning, grid) = placed(hw, problem.space(), levels, cube)?;
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

impl ReduceRoutine<CubeStagedReduceMicrokernel> for CubeStagedReduceRoutine {
    const NAME: &'static str = "cube-staged-reduce";

    type Out = TileTensorLaunch;

    fn out_arg(out: Bound) -> TileTensorLaunch {
        out.arg()
    }

    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        launcher: &Launcher,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: CubeStagedReduceBlueprint,
        grid: Grid,
    ) -> Result<(), Decline> {
        // SAFETY: every operand is bound through the launcher, which masks the reads and writes
        // of a tile the problem's edge cuts short, and the kernel indexes no buffer but through
        // its tiles; its shared memory is the slots its microkernel declares, which the grid
        // counted.
        unsafe {
            cube_staged_reduce_kernel::launch_unchecked::<F, I>(
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
    use crate::kernels::testing::{BufferLaunch, UnboundLaunch, launch_unbound};
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    prepares_more_row_axes_than_the_grid_holds!(CubeStagedReduceRoutine);
    declines_lacking_every_feature!(CubeStagedReduceRoutine);
    names_its_setting!(CubeStagedReduceRoutine, "planes=4 lane_slots=32");
    estimates_its_microkernel!(CubeStagedReduceRoutine, ReduceRoutineProblem::work_item);

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

    const LAUNCH: BufferLaunch = launch_first_served::<CubeStagedReduceRoutine, _>;
    const UNBOUND: UnboundLaunch = launch_unbound::<CubeStagedReduceRoutine, _>;

    reduces_each_dim!(LAUNCH, UNBOUND);
    reduces_more_row_axes_than_the_grid_has!(LAUNCH);
    reduces_with_indices!(LAUNCH);
}
