//! A candidate of the autotune, whichever routine it launches ([`ReduceCandidate`]).

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceTensors, UnitDirectReduceRoutine, tune::*,
};
use cubecl::Device;
use cubecl::client::Client;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Decline, TileOperand, Unbound};
use cubek_std::autotune::TuneCandidate;

/// A candidate of the autotune, whichever routine it launches: what the autotune identifies a
/// candidate by, and what the tuner hands back for the one it found fastest
/// ([`ReduceCandidate::fastest`]), so a launch outside a round, a fused one, runs the winner.
#[derive(Debug, Clone)]
pub struct ReduceCandidate {
    launcher: ReduceCandidateLauncher,
}

impl ReduceCandidate {
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

    /// The candidate, a routine at its setting, that the autotune found fastest for the
    /// reductions `key` describes on `client`'s `device`, or `None` where no round in this process
    /// has settled the key, or the floor named apart won it: what a fused reduce runs, so it
    /// inherits the kernel the unfused round chose.
    pub fn fastest(client: &Client, device: &Device, key: &ReduceKey) -> Option<Self> {
        fastest_tuned(client, device, key)
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
    UnitDirect(ReduceRoutineRunner<UnitDirectReduceRoutine>),
    PlaneShuffle(ReduceRoutineRunner<PlaneShuffleReduceRoutine>),
    CubeStaged(ReduceRoutineRunner<CubeStagedReduceRoutine>),
    HypercubeLastArrival(ReduceRoutineRunner<HypercubeLastArrivalReduceRoutine>),
    /// Run after the output is zeroed.
    HypercubeAtomic(ReduceRoutineRunner<HypercubeAtomicReduceRoutine>),
}

impl ReduceCandidateLauncher {
    /// `tensors`, whose input the caller binds, reduced at the candidate's setting
    /// ([`ReduceRoutineRunner::launch_unbound`]).
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
    fn name(&self) -> &str {
        match &self.dispatch {
            ReduceCandidateLauncherDispatch::UnitDirect(candidate) => candidate.strategy(),
            ReduceCandidateLauncherDispatch::PlaneShuffle(candidate) => candidate.strategy(),
            ReduceCandidateLauncherDispatch::CubeStaged(candidate) => candidate.strategy(),
            ReduceCandidateLauncherDispatch::HypercubeLastArrival(candidate) => {
                candidate.strategy()
            }
            ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate) => candidate.strategy(),
        }
    }
}

impl From<ReduceRoutineRunner<UnitDirectReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineRunner<UnitDirectReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::UnitDirect(candidate))
    }
}

impl From<ReduceRoutineRunner<PlaneShuffleReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineRunner<PlaneShuffleReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::PlaneShuffle(candidate))
    }
}

impl From<ReduceRoutineRunner<CubeStagedReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineRunner<CubeStagedReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::CubeStaged(candidate))
    }
}

impl From<ReduceRoutineRunner<HypercubeLastArrivalReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineRunner<HypercubeLastArrivalReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::HypercubeLastArrival(
            candidate,
        ))
    }
}

impl From<ReduceRoutineRunner<HypercubeAtomicReduceRoutine>> for ReduceCandidate {
    fn from(candidate: ReduceRoutineRunner<HypercubeAtomicReduceRoutine>) -> Self {
        Self::launching(ReduceCandidateLauncherDispatch::HypercubeAtomic(candidate))
    }
}
