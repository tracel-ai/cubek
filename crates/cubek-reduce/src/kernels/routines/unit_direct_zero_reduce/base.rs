use super::kernel::unit_direct_zero_reduce_kernel;
use crate::kernels::{
    ReduceBinding, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem, ReduceTensors,
    reduce_lane_counts, unit_launch,
};
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::launch::{Decline, TileOperand, Unbound, Vectorization};
use cubecl::tile::{Geometry, Launcher, TileTensorLaunch};
use cubek_microkernel::reduce::ReduceCarrierFamily;
use cubek_microkernel::zero::ZeroOperand;
use cubek_microkernel::zero::unit::{UnitDirectZeroMicrokernel, UnitDirectZeroMicrokernelSettings};
use cubek_std::Hardware;
use cubek_std::microkernel::{KernelEstimate, Microkernel};
use cubek_std::routine::{Grid, Routine};

/// A reduction's output zeroed, a unit per box of rows' cells: what a reduction that adds its
/// parts into the output starts from ([`HypercubeAtomicReduceRoutine`](crate::kernels::HypercubeAtomicReduceRoutine)).
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct UnitDirectZeroReduceRoutine;

/// The zero microkernel's setting, and how many planes each cube runs.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDirectZeroReduceSettings {
    pub microkernel: UnitDirectZeroMicrokernelSettings,
    pub plane_count: usize,
}

impl core::fmt::Display for UnitDirectZeroReduceSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{} planes={}", self.microkernel, self.plane_count)
    }
}

pub type UnitDirectZeroReduceBlueprint = ReduceBlueprint<UnitDirectZeroMicrokernel>;

impl Routine for UnitDirectZeroReduceRoutine {
    type Problem = ReduceRoutineProblem;
    type Microkernel = UnitDirectZeroMicrokernel;
    type Settings = UnitDirectZeroReduceSettings;
    type Blueprint = UnitDirectZeroReduceBlueprint;
    type Partitioning = ();

    /// Cubes of four planes: a zeroing is one write a unit, which no setting speeds up.
    fn settings(hw: &Hardware) -> Vec<Self::Settings> {
        UnitDirectZeroMicrokernel::settings(hw)
            .into_iter()
            .map(|microkernel| UnitDirectZeroReduceSettings {
                microkernel,
                plane_count: 4,
            })
            .collect()
    }

    fn vectorize(
        microkernel: &UnitDirectZeroMicrokernel,
        _partitioning: &(),
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
        let lane_counts = reduce_lane_counts(hw, problem, vectorization, None)?;
        let (partitioning, grid) = unit_launch(hw, problem, lane_counts[0], settings.plane_count)?;
        Ok((
            ReduceBlueprint::new(microkernel, partitioning, lane_counts),
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

impl ReduceRoutine<UnitDirectZeroMicrokernel> for UnitDirectZeroReduceRoutine {
    const NAME: &'static str = "unit-direct-zero-reduce";

    /// The output zeroed; the input is never read.
    fn dispatch<F: ReduceCarrierFamily, I: TileOperand>(
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
    ) {
        Self::zero_unbound(tensors, blueprint, grid);
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
    use crate::kernels::launch_first_served;
    use crate::kernels::testing::{Reduction, unwritten_output, written_output, zeros};
    use cubecl::zspace::Shape;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;
    use cubek_test_utils::{HostData, HostDataVec, ScopeUnderTest, Written};

    /// Zeroing reads no Feature.
    #[test]
    fn prepares_lacking_every_feature() {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        let problem = ReduceRoutineProblem::dense(&[4, 4096], 1, ReduceOperation::Sum);
        let settings = UnitDirectZeroReduceRoutine::settings(&hw)[0];
        let prepared = UnitDirectZeroReduceRoutine::prepare(&hw, &problem, &settings).map(|_| ());
        assert!(prepared.is_ok(), "{prepared:?}");
    }

    /// Launches the zero routine alone and checks that a reduction's output, which starts away from
    /// zero, holds only zeros after it.
    struct ZeroScopeUnderTest;

    impl ScopeUnderTest for ZeroScopeUnderTest {
        type Case = Reduction;

        fn launch(&self, client: &Client, reduction: &Reduction) -> Result<Vec<Written>, Decline> {
            let out = unwritten_output(client, reduction);
            let tensors = ReduceTensors::new(
                client,
                zeros(client, &reduction.shape),
                out.clone().binding(),
                reduction.axis,
                reduction.operation,
                reduction.elements(),
            );
            let hw = Hardware::from_client(client);
            launch_first_served::<UnitDirectZeroReduceRoutine, _>(&hw, &tensors)?;
            Ok(vec![written_output(out, reduction)])
        }

        /// Zeros, the output's shape.
        fn reference(&self, reduction: &Reduction) -> Vec<HostData> {
            let out_shape = reduction.out_shape();
            vec![HostData::row_major(
                HostDataVec::F32(vec![0.0; out_shape.iter().product()]),
                Shape::from(out_shape),
            )]
        }

        fn tolerance(&self, _reduction: &Reduction) -> f32 {
            0.0
        }
    }

    /// The output of `[5, 300, 7]` summed along its middle dim, 35 cells, all zeroed.
    #[test]
    fn zeroes_every_cell_of_the_output() {
        ZeroScopeUnderTest.check([Reduction {
            shape: vec![5, 300, 7],
            axis: 1,
            operation: ReduceOperation::Sum,
        }]);
    }
}
