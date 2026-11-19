use super::kernel::unit_direct_zero_reduce_kernel;
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem, ReduceTensors,
    reduce_widths,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Decline, TileOperand, Unbound, Vectorization};
use cubecl::tile::space::UnitCube;
use cubecl::tile::{Geometry, Launcher, Levels, Partitioning, TileTensorLaunch};
use cubek_microkernel::zero::ZeroOperand;
use cubek_microkernel::zero::unit::{UnitDirectZeroMicrokernel, UnitDirectZeroMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{CubeResources, Grid, Routine};

/// A reduction's output zeroed, a unit per box of rows' cells: what a reduction that adds its
/// parts into the output starts from ([`HypercubeAtomicReduceRoutine`](crate::kernels::HypercubeAtomicReduceRoutine)).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct UnitDirectZeroReduceRoutine;

/// The zero microkernel's setting, and how many planes each cube runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectZeroReduceSettings {
    pub microkernel: UnitDirectZeroMicrokernelSettings,
    pub planes: usize,
}

impl core::fmt::Display for UnitDirectZeroReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.planes)
    }
}

pub type UnitDirectZeroReduceBlueprint = ReduceBlueprint<UnitDirectZeroMicrokernel>;

impl Routine for UnitDirectZeroReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = UnitDirectZeroMicrokernel;
    type Settings = UnitDirectZeroReduceSettings;
    type Blueprint = UnitDirectZeroReduceBlueprint;
    type Partitioning = Partitioning;

    /// Cubes of four planes: a zeroing is one write a unit, which no setting speeds up.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        UnitDirectZeroMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| UnitDirectZeroReduceSettings {
                microkernel,
                planes: 4,
            })
            .collect()
    }

    fn vectorize(
        microkernel: &UnitDirectZeroMicrokernel,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<ZeroOperand>,
    ) {
        microkernel.vectorize(&(), vectorization);
    }

    /// Placed as the unit reduce places its owners, and its lines laid as a reduction lays them,
    /// so each unit zeroes the cells it would write.
    fn prepare(
        hw: &Hardware,
        problem: &ReduceRoutineProblem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Decline> {
        let microkernel = UnitDirectZeroMicrokernel::best(hw, &(), &settings.microkernel)?;
        let mut vectorization = Vectorization::new();
        vectorization.require(problem.work_item().requirement());
        // It zeroes a sum's output ahead of the atomic routine, which no top-k bounds.
        let widths = reduce_widths(hw, problem, vectorization, None);
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
        _problem: &ReduceRoutineProblem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    ) {
        blueprint
            .microkernel
            .estimate(&(), &(), grid.planes(), estimate);
    }
}

impl ReduceLauncher<UnitDirectZeroMicrokernel> for UnitDirectZeroReduceRoutine {
    const NAME: &'static str = "unit-direct-zero-reduce";

    /// The output zeroed; the input is never read.
    fn dispatch<I: TileOperand>(
        problem: &ReduceRoutineProblem,
        client: &Client,
        binding: ReduceBinding<<I::Arg as LaunchArg>::RuntimeArg, TileTensorLaunch>,
        blueprint: UnitDirectZeroReduceBlueprint,
        grid: Grid,
    ) {
        launch_zero(
            client,
            problem,
            &blueprint,
            binding.out,
            &binding.launcher,
            grid,
        );
    }

    /// The output zeroed. The caller's input is never read, so `input` is never called.
    fn launch_unbound<I: TileOperand>(
        tensors: &ReduceTensors<Geometry>,
        blueprint: UnitDirectZeroReduceBlueprint,
        grid: Grid,
        _input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        Self::zero_unbound(tensors, blueprint, grid);
        Ok(())
    }
}

impl UnitDirectZeroReduceRoutine {
    /// The output of `tensors`, whose input the caller binds, zeroed on `grid`.
    pub fn zero_unbound(
        tensors: &ReduceTensors<Geometry>,
        blueprint: UnitDirectZeroReduceBlueprint,
        grid: Grid,
    ) {
        let binding = tensors.bind(&blueprint, &grid);
        launch_zero(
            tensors.client(),
            tensors.problem(),
            &blueprint,
            binding.out.arg(),
            &binding.launcher,
            grid,
        );
    }
}

/// `out` zeroed on `grid` as `blueprint` states, through `launcher`'s partitioning.
fn launch_zero(
    client: &Client,
    problem: &ReduceRoutineProblem,
    blueprint: &UnitDirectZeroReduceBlueprint,
    out: TileTensorLaunch,
    launcher: &Launcher,
    grid: Grid,
) {
    // SAFETY: the output is bound through the launcher, which masks the writes of a tile the
    // problem's edge cuts short, and the kernel indexes no buffer but through its tile; it holds
    // no shared memory.
    unsafe {
        unit_direct_zero_reduce_kernel::launch_unchecked(
            client,
            grid.launch.cube_count,
            grid.launch.cube_dim,
            out,
            launcher.partitioning_arg(),
            blueprint.microkernel,
            problem.elements.out,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// tracel:kernel::routine-tested-without-features: zeroing reads none.
    #[test]
    fn prepares_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = UnitDirectZeroReduceRoutine::settings(&hw)[0];
        let prepared = UnitDirectZeroReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }
}
