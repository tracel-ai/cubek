//! A routine at one of its settings, whichever routine it is ([`ReduceCandidate`]), and the launch
//! it prepares for a key ([`ReducePrepared`]).

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceBlueprint, ReduceKey, ReduceRoutineSetting, ReduceTensors,
    UnitDirectReduceRoutine,
};
use cubecl::CubeDim;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Decline, TileOperand, Unbound};
use cubek_std::Hardware;
use cubek_std::routine::{Grid, LaunchSettings, Routine};

/// A routine at one of its settings, whichever routine it is: what a caller that measures the
/// routines identifies a candidate by, and runs again once it is found fastest, so a launch
/// outside a round, a fused one, runs the winner.
#[derive(Debug, Clone)]
pub struct ReduceCandidate {
    launcher: ReduceCandidateLauncher,
}

impl ReduceCandidate {
    /// Every setting of every routine `hw` lists, routine by routine: the unit, plane, cube,
    /// last-arrival hypercube and atomic hypercube routines, each in the order it lists its
    /// settings.
    pub fn list(hw: &Hardware) -> Vec<Self> {
        let unit = ReduceRoutineSetting::<UnitDirectReduceRoutine>::list(hw);
        let plane = ReduceRoutineSetting::<PlaneShuffleReduceRoutine>::list(hw);
        let cube = ReduceRoutineSetting::<CubeStagedReduceRoutine>::list(hw);
        let last_arrival = ReduceRoutineSetting::<HypercubeLastArrivalReduceRoutine>::list(hw);
        let atomic = ReduceRoutineSetting::<HypercubeAtomicReduceRoutine>::list(hw);
        unit.into_iter()
            .map(Self::from)
            .chain(plane.into_iter().map(Self::from))
            .chain(cube.into_iter().map(Self::from))
            .chain(last_arrival.into_iter().map(Self::from))
            .chain(atomic.into_iter().map(Self::from))
            .collect()
    }

    /// The routine's prefix and its setting, which a tuning cache is keyed by.
    pub fn name(&self) -> &String {
        self.launcher.name()
    }

    /// `tensors` reduced at this candidate's setting, the output zeroed first where the routine
    /// only adds into it.
    ///
    /// # Errors
    ///
    /// Why the routine, or the zeroing before it, cannot run the problem at its setting.
    ///
    /// # Panics
    ///
    /// As [`Reduce::execute`](crate::kernels::Reduce::execute).
    pub fn launch(&self, tensors: &ReduceTensors) -> Result<(), Decline> {
        self.launcher.launch(tensors)
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where its
    /// routine declines it on the device it was listed for.
    pub fn prepared(&self, key: &ReduceKey) -> Option<ReducePrepared> {
        self.launcher.prepared(key)
    }

    /// `tensors`, whose input the caller binds, reduced at this candidate's setting.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, or why it cannot read the caller's
    /// input.
    ///
    /// # Panics
    ///
    /// As [`Reduce::execute`](crate::kernels::Reduce::execute).
    pub fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        self.launcher.launch_unbound::<I>(tensors, input)
    }
}

impl ReduceCandidate {
    /// The candidate that launches through `dispatch`.
    fn launching(dispatch: ReduceCandidateLauncherDispatch) -> Self {
        Self {
            launcher: ReduceCandidateLauncher { dispatch },
        }
    }
}

/// The candidate's name: its routine's prefix and setting, which the tuning cache is keyed by.
impl core::fmt::Display for ReduceCandidate {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(self.launcher.name())
    }
}

/// Launches a [`ReduceCandidate`] through whichever routine it holds.
#[derive(Debug, Clone)]
struct ReduceCandidateLauncher {
    dispatch: ReduceCandidateLauncherDispatch,
}

/// The routine a [`ReduceCandidateLauncher`] launches through, one variant a routine.
#[derive(Debug, Clone)]
enum ReduceCandidateLauncherDispatch {
    UnitDirect(ReduceRoutineSetting<UnitDirectReduceRoutine>),
    PlaneShuffle(ReduceRoutineSetting<PlaneShuffleReduceRoutine>),
    CubeStaged(ReduceRoutineSetting<CubeStagedReduceRoutine>),
    HypercubeLastArrival(ReduceRoutineSetting<HypercubeLastArrivalReduceRoutine>),
    /// Run after the output is zeroed.
    HypercubeAtomic(ReduceRoutineSetting<HypercubeAtomicReduceRoutine>),
}

impl ReduceCandidateLauncher {
    /// `tensors` reduced at the candidate's setting, zeroed first for the atomic routine.
    fn launch(&self, tensors: &ReduceTensors) -> Result<(), Decline> {
        match &self.dispatch {
            ReduceCandidateLauncherDispatch::UnitDirect(candidate) => candidate.launch(tensors),
            ReduceCandidateLauncherDispatch::PlaneShuffle(candidate) => candidate.launch(tensors),
            ReduceCandidateLauncherDispatch::CubeStaged(candidate) => candidate.launch(tensors),
            ReduceCandidateLauncherDispatch::HypercubeLastArrival(candidate) => {
                candidate.launch(tensors)
            }
            ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate) => {
                candidate.launch_zeroed(tensors)
            }
        }
    }

    /// The launch the candidate prepares for `key`, its blueprint tagged with its routine.
    fn prepared(&self, key: &ReduceKey) -> Option<ReducePrepared> {
        match &self.dispatch {
            ReduceCandidateLauncherDispatch::UnitDirect(candidate) => {
                candidate.prepared(key).map(|(blueprint, grid)| {
                    ReducePrepared::new(ReduceCompiled::UnitDirect, blueprint, grid)
                })
            }
            ReduceCandidateLauncherDispatch::PlaneShuffle(candidate) => {
                candidate.prepared(key).map(|(blueprint, grid)| {
                    ReducePrepared::new(ReduceCompiled::PlaneShuffle, blueprint, grid)
                })
            }
            ReduceCandidateLauncherDispatch::CubeStaged(candidate) => {
                candidate.prepared(key).map(|(blueprint, grid)| {
                    ReducePrepared::new(ReduceCompiled::CubeStaged, blueprint, grid)
                })
            }
            ReduceCandidateLauncherDispatch::HypercubeLastArrival(candidate) => {
                candidate.prepared(key).map(|(blueprint, grid)| {
                    ReducePrepared::new(ReduceCompiled::HypercubeLastArrival, blueprint, grid)
                })
            }
            ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate) => {
                candidate.prepared(key).map(|(blueprint, grid)| {
                    ReducePrepared::new(ReduceCompiled::HypercubeAtomic, blueprint, grid)
                })
            }
        }
    }

    /// `tensors`, whose input the caller binds, reduced at the candidate's setting
    /// ([`ReduceRoutineSetting::launch_unbound`]).
    fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        match &self.dispatch {
            ReduceCandidateLauncherDispatch::UnitDirect(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::PlaneShuffle(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::CubeStaged(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::HypercubeLastArrival(candidate) => {
                candidate.launch_unbound::<I>(tensors, input)
            }
            ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate) => {
                candidate.launch_zeroed_unbound::<I>(tensors, input)
            }
        }
    }

    /// The candidate's name, which the tuning cache is keyed by.
    fn name(&self) -> &String {
        match &self.dispatch {
            ReduceCandidateLauncherDispatch::UnitDirect(candidate) => candidate.name(),
            ReduceCandidateLauncherDispatch::PlaneShuffle(candidate) => candidate.name(),
            ReduceCandidateLauncherDispatch::CubeStaged(candidate) => candidate.name(),
            ReduceCandidateLauncherDispatch::HypercubeLastArrival(candidate) => candidate.name(),
            ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate) => candidate.name(),
        }
    }
}

impl From<ReduceRoutineSetting<UnitDirectReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineSetting<UnitDirectReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::UnitDirect(candidate))
    }
}

impl From<ReduceRoutineSetting<PlaneShuffleReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineSetting<PlaneShuffleReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::PlaneShuffle(candidate))
    }
}

impl From<ReduceRoutineSetting<CubeStagedReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineSetting<CubeStagedReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::CubeStaged(candidate))
    }
}

impl From<ReduceRoutineSetting<HypercubeLastArrivalReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineSetting<HypercubeLastArrivalReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::HypercubeLastArrival(
            candidate,
        ))
    }
}

impl From<ReduceRoutineSetting<HypercubeAtomicReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineSetting<HypercubeAtomicReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate))
    }
}

/// The launch a [`ReduceCandidate`] prepares for a key: the grid it runs on, and what it compiles,
/// which two candidates compile alike exactly where their prepared launches are equal.
#[derive(Clone)]
pub struct ReducePrepared {
    compiled: ReduceCompiled,
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
    fn new<M>(
        routine: fn(ReduceBlueprint<M>, CubeDim) -> ReduceCompiled,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    ) -> Self {
        Self {
            compiled: routine(blueprint, grid.launch.cube_dim),
            launch: grid.launch,
        }
    }
}

/// Two launches compile alike where their routine, blueprint and cube do: two settings preparing
/// one blueprint can still launch different cubes, and the count of cubes is a runtime value.
impl PartialEq for ReducePrepared {
    fn eq(&self, other: &Self) -> bool {
        self.compiled == other.compiled
    }
}

/// What a prepared launch compiles: its routine, its blueprint and its cube.
#[derive(Clone, PartialEq)]
enum ReduceCompiled {
    UnitDirect(
        ReduceBlueprint<<UnitDirectReduceRoutine as Routine>::Microkernel>,
        CubeDim,
    ),
    PlaneShuffle(
        ReduceBlueprint<<PlaneShuffleReduceRoutine as Routine>::Microkernel>,
        CubeDim,
    ),
    CubeStaged(
        ReduceBlueprint<<CubeStagedReduceRoutine as Routine>::Microkernel>,
        CubeDim,
    ),
    HypercubeLastArrival(
        ReduceBlueprint<<HypercubeLastArrivalReduceRoutine as Routine>::Microkernel>,
        CubeDim,
    ),
    HypercubeAtomic(
        ReduceBlueprint<<HypercubeAtomicReduceRoutine as Routine>::Microkernel>,
        CubeDim,
    ),
}
