//! What a caller can ask the reduce family for.
//!
//! Every variant but `Specific` is a policy; `Specific` is a pin, as in the matmul and attention
//! families.

use crate::ReduceError;
use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceBlueprint, HypercubeAtomicReduceRoutine,
    HypercubeLastArrivalReduceRoutine, PlaneShuffleReduceRoutine, ReduceRoutine, ReduceTensors,
    UnitDirectReduceRoutine, UnitDirectZeroReduceRoutine, launch_tuned, select,
};
use cubecl::Device;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Decline, TileOperand, Unbound};
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};
use serde::{Deserialize, Serialize};

/// How a reduction picks its kernel: measure the routines, pick one by rule, or run one named
/// kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ReduceOpStrategy {
    /// Pick by measurement: every routine at every setting the device lists joins an autotune, and
    /// the fastest runs, per problem and per device. The default.
    #[default]
    Autotune,
    /// Pick by rule, reaching no further than the named family.
    Heuristic(ReduceKernelHeuristic),
    /// Run exactly this kernel, for measuring it or holding it to a reference. A kernel that
    /// cannot run the problem returns its decline, never another kernel's result.
    Specific(ReduceKernel),
}

/// How far a [`Heuristic`](ReduceOpStrategy::Heuristic) may reach. Nothing here measures.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceKernelHeuristic {
    /// The routines in [`kernels`](crate::kernels), by rule.
    Tile,
}

/// The kernels themselves, named for their scope and how their partial results meet: the
/// vocabulary a [`Specific`](ReduceOpStrategy::Specific) pin speaks. Each runs the first of its
/// routine's settings that serves the problem.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceKernel {
    /// `UnitDirectReduceRoutine`: a unit reads a whole row. Asks nothing of the device, so it serves
    /// every problem everywhere.
    Unit,
    /// `PlaneShuffleReduceRoutine`: a plane reads a row, its lanes' partials meeting through plane
    /// instructions.
    PlaneShuffle,
    /// `CubeStagedReduceRoutine`: a cube reads a row, its planes' partials meeting in shared memory.
    CubeStaged,
    /// `HypercubeLastArrivalReduceRoutine`: a row split across cubes, the last cube to arrive merging
    /// every part.
    HypercubeLastArrival,
    /// `HypercubeAtomicReduceRoutine` after `UnitDirectZeroReduceRoutine`: a row split across cubes, each
    /// adding its part into the zeroed output. Sums only.
    HypercubeAtomic,
}

impl ReduceOpStrategy {
    /// Every strategy: the policies, then a pin of every kernel.
    pub const ALL: [Self; 7] = [
        Self::Autotune,
        Self::Heuristic(ReduceKernelHeuristic::Tile),
        Self::Specific(ReduceKernel::Unit),
        Self::Specific(ReduceKernel::PlaneShuffle),
        Self::Specific(ReduceKernel::CubeStaged),
        Self::Specific(ReduceKernel::HypercubeLastArrival),
        Self::Specific(ReduceKernel::HypercubeAtomic),
    ];

    /// Whether this strategy lets the caller pick the kernel, rather than naming one.
    pub fn is_policy(&self) -> bool {
        !matches!(self, Self::Specific(_))
    }

    /// What a report calls this strategy, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Autotune => "autotune",
            Self::Heuristic(heuristic) => heuristic.name(),
            Self::Specific(kernel) => kernel.name(),
        }
    }
}

impl ReduceKernelHeuristic {
    /// Every ceiling.
    pub const ALL: [Self; 1] = [Self::Tile];

    /// What a report calls this ceiling, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tile => "tile",
        }
    }
}

impl ReduceKernel {
    /// Every kernel a pin names.
    pub const ALL: [Self; 5] = [
        Self::Unit,
        Self::PlaneShuffle,
        Self::CubeStaged,
        Self::HypercubeLastArrival,
        Self::HypercubeAtomic,
    ];

    /// What a report calls this kernel, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Unit => "unit",
            Self::PlaneShuffle => "plane-shuffle",
            Self::CubeStaged => "cube-staged",
            Self::HypercubeLastArrival => "hypercube-last-arrival",
            Self::HypercubeAtomic => "hypercube-atomic",
        }
    }
}

/// `tensors` reduced on `device` by the kernel `strategy` picks.
///
/// # Errors
///
/// [`ReduceError::Declined`] where the picked kernel cannot run the problem on this device.
pub(crate) fn launch_by_strategy(
    strategy: ReduceOpStrategy,
    device: &Device,
    tensors: ReduceTensors,
) -> Result<(), ReduceError> {
    let hw = Hardware::from_client(tensors.client());
    match strategy {
        ReduceOpStrategy::Autotune => launch_tuned(tensors, device),
        ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile) => Ok(select(&hw, &tensors)?),
        ReduceOpStrategy::Specific(kernel) => Ok(launch_kernel(kernel, &hw, &tensors)?),
    }
}

/// `tensors` reduced by `kernel` on `hw`.
///
/// # Errors
///
/// Why its routine cannot run the problem here.
pub(crate) fn launch_kernel(
    kernel: ReduceKernel,
    hw: &Hardware,
    tensors: &ReduceTensors,
) -> Result<(), Decline> {
    match kernel {
        ReduceKernel::Unit => launch_first_served::<UnitDirectReduceRoutine, _>(hw, tensors),
        ReduceKernel::PlaneShuffle => {
            launch_first_served::<PlaneShuffleReduceRoutine, _>(hw, tensors)
        }
        ReduceKernel::CubeStaged => launch_first_served::<CubeStagedReduceRoutine, _>(hw, tensors),
        ReduceKernel::HypercubeLastArrival => {
            launch_first_served::<HypercubeLastArrivalReduceRoutine, _>(hw, tensors)
        }
        ReduceKernel::HypercubeAtomic => launch_zeroed_atomic(
            hw,
            tensors,
            HypercubeAtomicReduceRoutine::prepare_first_served(hw, tensors.problem())?,
        ),
    }
}

/// `tensors` summed by the atomic hypercube, at the launch `prepared` holds, into their output,
/// zeroed first. The zeroing prepares before either launches, so a decline leaves the output as
/// the caller gave it.
///
/// # Errors
///
/// Why the zeroing cannot run the problem here.
pub(crate) fn launch_zeroed_atomic(
    hw: &Hardware,
    tensors: &ReduceTensors,
    prepared: (HypercubeAtomicReduceBlueprint, Grid),
) -> Result<(), Decline> {
    let (zero_blueprint, zero_grid) =
        UnitDirectZeroReduceRoutine::prepare_first_served(hw, tensors.problem())?;
    let (blueprint, grid) = prepared;
    UnitDirectZeroReduceRoutine::launch(tensors, zero_blueprint, zero_grid);
    HypercubeAtomicReduceRoutine::launch(tensors, blueprint, grid);
    Ok(())
}

/// [`launch_zeroed_atomic`] on an input the caller binds through `input`.
///
/// # Errors
///
/// As [`launch_zeroed_atomic`].
pub(crate) fn launch_zeroed_atomic_unbound<I: TileOperand>(
    hw: &Hardware,
    tensors: &ReduceTensors<Geometry>,
    prepared: (HypercubeAtomicReduceBlueprint, Grid),
    input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
) -> Result<(), Decline> {
    let (zero_blueprint, zero_grid) =
        UnitDirectZeroReduceRoutine::prepare_first_served(hw, tensors.problem())?;
    let (blueprint, grid) = prepared;
    UnitDirectZeroReduceRoutine::zero_unbound(tensors, zero_blueprint, zero_grid);
    HypercubeAtomicReduceRoutine::launch_unbound::<I>(tensors, blueprint, grid, input);
    Ok(())
}

/// `tensors` reduced by `R` at the first of its settings `hw` prepares them at.
///
/// # Errors
///
/// Why none prepares them ([`Routine::prepare_first_served`]).
pub(crate) fn launch_first_served<R: ReduceRoutine<M>, M>(
    hw: &Hardware,
    tensors: &ReduceTensors,
) -> Result<(), Decline> {
    let (blueprint, grid) = R::prepare_first_served(hw, tensors.problem())?;
    R::launch(tensors, blueprint, grid);
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::BufferLaunch;

    #[test]
    fn the_autotune_comes_first() {
        assert_eq!(ReduceOpStrategy::ALL[0], ReduceOpStrategy::Autotune);
    }

    /// The policies, the autotune and each heuristic, come before the pins.
    #[test]
    fn the_policies_come_before_the_pins() {
        let policies = ReduceOpStrategy::ALL
            .iter()
            .take_while(|strategy| strategy.is_policy())
            .count();
        assert_eq!(policies, 1 + ReduceKernelHeuristic::ALL.len());
    }

    #[test]
    fn every_kernel_is_pinned() {
        let pins = ReduceOpStrategy::ALL
            .into_iter()
            .filter(|strategy| !strategy.is_policy())
            .count();
        assert_eq!(pins, ReduceKernel::ALL.len());
    }

    #[test]
    fn every_name_is_kebab_case() {
        let kebab = |name: &str| name.chars().all(|c| c.is_ascii_lowercase() || c == '-');
        assert!(
            ReduceOpStrategy::ALL
                .iter()
                .all(|strategy| kebab(strategy.name()))
        );
    }

    #[test]
    fn every_name_is_its_own() {
        let all = ReduceOpStrategy::ALL;
        let distinct: std::collections::HashSet<&str> =
            all.iter().map(ReduceOpStrategy::name).collect();
        assert_eq!(distinct.len(), all.len());
    }

    /// The unit kernel, pinned.
    mod unit {
        use super::*;

        const LAUNCH: BufferLaunch = |hw, tensors| launch_kernel(ReduceKernel::Unit, hw, tensors);

        strategy_reduces_each_problem!(LAUNCH);
    }

    /// The plane kernel, pinned.
    mod plane_shuffle {
        use super::*;

        const LAUNCH: BufferLaunch =
            |hw, tensors| launch_kernel(ReduceKernel::PlaneShuffle, hw, tensors);

        strategy_reduces_each_problem!(LAUNCH);
    }

    /// The cube kernel, pinned.
    mod cube_staged {
        use super::*;

        const LAUNCH: BufferLaunch =
            |hw, tensors| launch_kernel(ReduceKernel::CubeStaged, hw, tensors);

        strategy_reduces_each_problem!(LAUNCH);
    }

    /// The last-arrival hypercube kernel, pinned.
    mod hypercube_last_arrival {
        use super::*;

        const LAUNCH: BufferLaunch =
            |hw, tensors| launch_kernel(ReduceKernel::HypercubeLastArrival, hw, tensors);

        strategy_reduces_each_problem!(LAUNCH);
    }
}
