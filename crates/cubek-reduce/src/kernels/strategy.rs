//! What a caller can ask the reduce family for.
//!
//! Every variant but `Specific` is a policy; `Specific` is a pin, as in the matmul and attention
//! families.

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceLauncher, ReduceTensors, UnitDirectReduceRoutine,
    UnitDirectZeroRoutine,
};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use serde::{Deserialize, Serialize};

/// How a reduction picks its kernel: measure the routines, pick one by rule, or run one named
/// kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
pub enum ReduceOpStrategy {
    /// Pick by measurement: every routine at every setting the device lists joins an autotune
    /// ([`tune`](super::tune)), and the fastest runs, per problem and per device. The default.
    #[default]
    Autotune,
    /// Pick by rule, reaching no further than the named family ([`select`](super::select)).
    Heuristic(ReduceKernelHeuristic),
    /// Run exactly this kernel, for measuring it or holding it to a reference. A kernel that
    /// cannot run the problem returns its refusal, never another kernel's result.
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
    /// [`UnitDirectReduceRoutine`]: a unit reads a whole row. Asks nothing of the device, so it serves
    /// every problem everywhere.
    Unit,
    /// [`PlaneShuffleReduceRoutine`]: a plane reads a row, its lanes' partials meeting through plane
    /// instructions.
    PlaneShuffle,
    /// [`CubeStagedReduceRoutine`]: a cube reads a row, its planes' partials meeting in shared memory.
    CubeStaged,
    /// [`HypercubeLastArrivalReduceRoutine`]: a row split across cubes, the last cube to arrive merging
    /// every part.
    HypercubeLastArrival,
    /// [`HypercubeAtomicReduceRoutine`] after [`UnitDirectZeroRoutine`]: a row split across cubes, each
    /// adding its part into the zeroed output. Sums only.
    HypercubeAtomic,
}

impl ReduceOpStrategy {
    /// Every strategy: the policies, then a pin of every kernel.
    pub fn all() -> Vec<Self> {
        let heuristics = ReduceKernelHeuristic::all()
            .into_iter()
            .map(Self::Heuristic);
        let pins = ReduceKernel::all().into_iter().map(Self::Specific);
        core::iter::once(Self::Autotune)
            .chain(heuristics)
            .chain(pins)
            .collect()
    }

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
    pub fn all() -> Vec<Self> {
        vec![Self::Tile]
    }

    /// What a report calls this ceiling, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Tile => "tile",
        }
    }
}

impl ReduceKernel {
    /// Every kernel a pin names.
    pub fn all() -> Vec<Self> {
        vec![
            Self::Unit,
            Self::PlaneShuffle,
            Self::CubeStaged,
            Self::HypercubeLastArrival,
            Self::HypercubeAtomic,
        ]
    }

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

    /// `tensors` reduced by this kernel on `hw`.
    ///
    /// # Errors
    ///
    /// Why its routine cannot run the problem here.
    pub fn launch(&self, hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Refusal> {
        match self {
            Self::Unit => launch_first_served::<UnitDirectReduceRoutine, _>(hw, tensors),
            Self::PlaneShuffle => launch_first_served::<PlaneShuffleReduceRoutine, _>(hw, tensors),
            Self::CubeStaged => launch_first_served::<CubeStagedReduceRoutine, _>(hw, tensors),
            Self::HypercubeLastArrival => {
                launch_first_served::<HypercubeLastArrivalReduceRoutine, _>(hw, tensors)
            }
            Self::HypercubeAtomic => {
                launch_first_served::<UnitDirectZeroRoutine, _>(hw, tensors)?;
                launch_first_served::<HypercubeAtomicReduceRoutine, _>(hw, tensors)
            }
        }
    }
}

/// `tensors` reduced by `R` at the first of its settings `hw` prepares them at.
fn launch_first_served<R: ReduceLauncher<M>, M>(
    hw: &Hardware,
    tensors: &ReduceTensors,
) -> Result<(), Refusal> {
    let (blueprint, grid) = R::prepare_first_served(hw, tensors.problem())?;
    R::launch(tensors, blueprint, grid);
    Ok(())
}
