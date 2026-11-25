//! What a caller can ask the reduce family for.
//!
//! Every variant but `Specific` is a policy; `Specific` is a pin, as in the matmul and attention
//! families. None measures: a caller that measures the routines races their settings itself
//! ([`ReduceCandidate`](crate::kernels::ReduceCandidate)).

use crate::ReduceError;
use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceBlueprint, HypercubeAtomicReduceRoutine,
    HypercubeLastArrivalReduceRoutine, PlaneShuffleReduceRoutine, ReduceRoutine, ReduceTensors,
    UnitDirectReduceRoutine, UnitDirectZeroReduceRoutine, select,
};
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Decline, TileOperand, Unbound};
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};
use serde::{Deserialize, Serialize};

/// How a reduction picks its kernel: pick one by rule, or run one named kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReduceOpStrategy {
    /// Pick by rule, reaching no further than the named family. The default, on the tile routines.
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
    pub const ALL: [Self; 6] = [
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
            Self::Heuristic(heuristic) => heuristic.name(),
            Self::Specific(kernel) => kernel.name(),
        }
    }
}

impl Default for ReduceOpStrategy {
    fn default() -> Self {
        Self::Heuristic(ReduceKernelHeuristic::Tile)
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

/// `tensors` reduced by the kernel `strategy` picks.
///
/// # Errors
///
/// [`ReduceError::Declined`] where the picked kernel cannot run the problem on this device.
pub(crate) fn launch_by_strategy(
    strategy: ReduceOpStrategy,
    tensors: ReduceTensors,
) -> Result<(), ReduceError> {
    let hw = Hardware::from_client(tensors.client());
    match strategy {
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
    UnitDirectZeroReduceRoutine::launch(tensors, zero_blueprint, zero_grid);
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
    use crate::kernels::testing::{
        BufferLaunch, BufferScopeChecker, Reduction, UnboundScopeChecker, bound_back,
        sum_of_long_rows, unbound_tensors,
    };
    use cubecl::client::Client;
    use cubecl::prelude::TensorBinding;
    use cubecl::tile::launch::StoredTileOperand;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_test_utils::ScopeChecker;

    #[test]
    fn the_default_is_the_rule() {
        assert_eq!(
            ReduceOpStrategy::default(),
            ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile)
        );
    }

    /// The policies, each heuristic, come before the pins.
    #[test]
    fn the_policies_come_before_the_pins() {
        let policies = ReduceOpStrategy::ALL
            .iter()
            .take_while(|strategy| strategy.is_policy())
            .count();
        assert_eq!(policies, ReduceKernelHeuristic::ALL.len());
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

    /// `tensors` reduced by the kernel `strategy` picks.
    fn launches_by(strategy: ReduceOpStrategy) -> Result<(), ReduceError> {
        let tensors = ReduceTensors::zeros(&[4, 300], 1, ReduceOperation::Max);
        launch_by_strategy(strategy, tensors)
    }

    #[test]
    fn the_rule_launches() {
        let rule = ReduceOpStrategy::Heuristic(ReduceKernelHeuristic::Tile);
        assert!(launches_by(rule).is_ok());
    }

    /// The atomic hypercube sums only.
    #[test]
    fn a_pinned_kernel_declines_what_it_does_not_run() {
        let launched = launches_by(ReduceOpStrategy::Specific(ReduceKernel::HypercubeAtomic));
        assert!(
            matches!(launched, Err(ReduceError::Declined(_))),
            "{launched:?}"
        );
    }

    /// The atomic hypercube launched on `tensors` into their output, zeroed first.
    ///
    /// # Errors
    ///
    /// Why it runs no such reduction here.
    fn launch_zeroed(hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Decline> {
        let prepared = HypercubeAtomicReduceRoutine::prepare_first_served(hw, tensors.problem())?;
        launch_zeroed_atomic(hw, tensors, prepared)
    }

    /// [`launch_zeroed`] of an input the caller binds.
    ///
    /// # Errors
    ///
    /// As [`launch_zeroed`].
    fn launch_zeroed_unbound(
        client: &Client,
        input: TensorBinding,
        out: TensorBinding,
        reduction: &Reduction,
    ) -> Result<(), Decline> {
        let tensors =
            unbound_tensors::<HypercubeAtomicReduceRoutine, _>(client, &input, out, reduction)?;
        let hw = Hardware::from_client(client);
        let prepared = HypercubeAtomicReduceRoutine::prepare_first_served(&hw, tensors.problem())?;
        launch_zeroed_atomic_unbound::<StoredTileOperand>(
            &hw,
            &tensors,
            prepared,
            bound_back(input),
        )
    }

    /// An output that starts away from zero holds the sum alone.
    #[test]
    fn the_atomic_hypercube_sums_into_a_zeroed_output() {
        BufferScopeChecker::new(launch_zeroed).check([sum_of_long_rows()]);
    }

    #[test]
    fn the_atomic_hypercube_sums_an_unbound_input_into_a_zeroed_output() {
        UnboundScopeChecker::new(launch_zeroed_unbound).check([sum_of_long_rows()]);
    }
}
