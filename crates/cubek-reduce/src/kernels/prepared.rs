//! The launch a [`ReduceCandidate`](crate::kernels::ReduceCandidate) prepares for a key
//! ([`ReducePrepared`]).

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceBlueprint, UnitDirectReduceRoutine,
};
use cubek_std::routine::{Grid, LaunchSettings, Routine};

/// The launch a [`ReduceCandidate`](crate::kernels::ReduceCandidate) prepares for a key: the grid it
/// runs on, and what it compiles, which two candidates compile alike exactly where their prepared
/// launches are equal.
#[derive(Clone)]
pub struct ReducePrepared {
    blueprint: ReduceRoutineBlueprint,
    launch: LaunchSettings,
}

impl ReducePrepared {
    /// The cubes and units the launch spawns.
    pub fn launch(&self) -> &LaunchSettings {
        &self.launch
    }
}

impl ReducePrepared {
    /// The launch of `blueprint` on `grid`, by the routine `routine` tags it with.
    pub(crate) fn new<M>(
        routine: fn(ReduceBlueprint<M>) -> ReduceRoutineBlueprint,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    ) -> Self {
        Self {
            blueprint: routine(blueprint),
            launch: grid.launch,
        }
    }
}

/// Two launches compile alike where their routine and blueprint do: the blueprint's partitioning
/// settles the cube on the device it was listed for, and the count of cubes is a runtime value.
impl PartialEq for ReducePrepared {
    fn eq(&self, other: &Self) -> bool {
        self.blueprint == other.blueprint
    }
}

/// What a prepared launch compiles: a blueprint of one of the routines, tagged with it.
#[derive(Clone, PartialEq)]
pub(crate) enum ReduceRoutineBlueprint {
    UnitDirect(ReduceBlueprint<<UnitDirectReduceRoutine as Routine>::Microkernel>),
    PlaneShuffle(ReduceBlueprint<<PlaneShuffleReduceRoutine as Routine>::Microkernel>),
    CubeStaged(ReduceBlueprint<<CubeStagedReduceRoutine as Routine>::Microkernel>),
    HypercubeLastArrival(
        ReduceBlueprint<<HypercubeLastArrivalReduceRoutine as Routine>::Microkernel>,
    ),
    HypercubeAtomic(ReduceBlueprint<<HypercubeAtomicReduceRoutine as Routine>::Microkernel>),
}
