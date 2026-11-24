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
    /// reductions `key` describes on `client`'s `device`, or `None` where no reduction in this
    /// process has settled the key, by a round or from the winner one left on disk, or the floor
    /// named apart won it: what a fused reduce runs, so it
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
    pub fn launch_unbound<I: TileOperand>(
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
    pub fn name(&self) -> &str {
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::testing::{
        Reduction, bound_back, reduces_unbound_like_the_cpu, sum_of_long_rows, unbound_tensors,
    };
    use crate::kernels::{ReduceElements, ReduceRoutine, ReduceRoutineProblem, launch_tuned};
    use cubecl::ir::{ElemType, FloatKind};
    use cubecl::prelude::TensorBinding;
    use cubecl::tile::launch::StoredTileOperand;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::routine::Routine;
    use cubek_std::{Hardware, devices};

    const F32: ElemType = ElemType::Float(FloatKind::F32);
    const SHAPE: [usize; 2] = [4, 8];

    /// Checks that `candidate` made a [`ReduceCandidate`] keeps its name.
    fn keeps_its_name<R: Routine>(candidate: &ReduceRoutineRunner<R>)
    where
        ReduceRoutineRunner<R>: TuneCandidate<Strategy = String> + Clone,
        ReduceCandidate: From<ReduceRoutineRunner<R>>,
    {
        let named = ReduceCandidate::from(candidate.clone()).to_string();
        assert_eq!(&named, candidate.strategy());
    }

    #[test]
    fn a_unit_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineRunner::<UnitDirectReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn a_plane_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineRunner::<PlaneShuffleReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn a_cube_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineRunner::<CubeStagedReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn a_last_arrival_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineRunner::<HypercubeLastArrivalReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn an_atomic_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineRunner::<HypercubeAtomicReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    /// Checks that `candidate` declines a sum given no scratch on a device lacking every Feature,
    /// without asking for its input.
    fn declines_unbound(candidate: impl Into<ReduceCandidate>) {
        let tensors = ReduceTensors::unbound_zeros(&SHAPE, 1, ReduceOperation::Sum);
        let launched = candidate
            .into()
            .launch_unbound::<StoredTileOperand>(&tensors, |_| {
                unreachable!("a declined candidate binds no input")
            });
        assert!(launched.is_err());
    }

    fn lacking_every_feature() -> Hardware {
        devices::lacking_every_feature(devices::m2_pro())
    }

    #[test]
    fn a_plane_candidate_declines_an_unbound_input() {
        declines_unbound(
            ReduceRoutineRunner::<PlaneShuffleReduceRoutine>::list(&lacking_every_feature())
                .remove(0),
        );
    }

    #[test]
    fn a_cube_candidate_declines_an_unbound_input() {
        declines_unbound(
            ReduceRoutineRunner::<CubeStagedReduceRoutine>::list(&lacking_every_feature())
                .remove(0),
        );
    }

    #[test]
    fn a_last_arrival_candidate_declines_an_unbound_input() {
        declines_unbound(
            ReduceRoutineRunner::<HypercubeLastArrivalReduceRoutine>::list(
                &lacking_every_feature(),
            )
            .remove(0),
        );
    }

    #[test]
    fn an_atomic_candidate_declines_an_unbound_input() {
        declines_unbound(
            ReduceRoutineRunner::<HypercubeAtomicReduceRoutine>::list(&lacking_every_feature())
                .remove(0),
        );
    }

    /// `reduction` launched unbound by `R`'s first candidate on `client`'s device.
    ///
    /// # Errors
    ///
    /// Why the candidate declines it.
    fn launch_candidate<R, M>(
        client: &Client,
        input: TensorBinding,
        out: TensorBinding,
        reduction: &Reduction,
    ) -> Result<(), Decline>
    where
        R: ReduceRoutine<M> + Routine<Microkernel = M, Settings: core::fmt::Display>,
        ReduceCandidate: From<ReduceRoutineRunner<R>>,
    {
        let hw = Hardware::from_client(client);
        let tensors = unbound_tensors::<R, M>(client, &input, out, reduction)?;
        let candidate = ReduceCandidate::from(ReduceRoutineRunner::<R>::list(&hw).remove(0));
        candidate.launch_unbound::<StoredTileOperand>(&tensors, bound_back(input))
    }

    #[test]
    fn a_unit_candidate_reduces_the_input_its_caller_binds() {
        reduces_unbound_like_the_cpu(
            launch_candidate::<UnitDirectReduceRoutine, _>,
            sum_of_long_rows(),
        );
    }

    #[test]
    fn a_plane_candidate_reduces_the_input_its_caller_binds() {
        reduces_unbound_like_the_cpu(
            launch_candidate::<PlaneShuffleReduceRoutine, _>,
            sum_of_long_rows(),
        );
    }

    #[test]
    fn a_cube_candidate_reduces_the_input_its_caller_binds() {
        reduces_unbound_like_the_cpu(
            launch_candidate::<CubeStagedReduceRoutine, _>,
            sum_of_long_rows(),
        );
    }

    #[test]
    fn a_last_arrival_candidate_reduces_the_input_its_caller_binds() {
        reduces_unbound_like_the_cpu(
            launch_candidate::<HypercubeLastArrivalReduceRoutine, _>,
            sum_of_long_rows(),
        );
    }

    #[test]
    fn an_atomic_candidate_reduces_the_input_its_caller_binds() {
        reduces_unbound_like_the_cpu(
            launch_candidate::<HypercubeAtomicReduceRoutine, _>,
            sum_of_long_rows(),
        );
    }

    /// A round settles a key on the candidate it found fastest.
    #[test]
    fn a_settled_key_has_a_fastest_candidate() {
        let device = cubecl::test_device();
        let tensors = ReduceTensors::zeros(&[3, 8], 1, ReduceOperation::Sum);
        let key = ReduceKey::from(&tensors);
        launch_tuned(tensors, &device).expect("the floor serves every problem");
        assert!(ReduceCandidate::fastest(&device.client(), &device, &key).is_some());
    }

    /// A sum no caller asks for: it accumulates in less than it reads.
    #[test]
    fn a_key_no_reduction_settled_has_no_fastest_candidate() {
        let device = cubecl::test_device();
        let problem = ReduceRoutineProblem {
            elements: ReduceElements {
                input: ElemType::Float(FloatKind::F64),
                accumulator: ElemType::Float(FloatKind::F16),
                out: F32,
            },
            ..ReduceRoutineProblem::dense(&SHAPE, 1, ReduceOperation::Sum)
        };
        let key = ReduceKey::from(&problem);
        assert!(ReduceCandidate::fastest(&device.client(), &device, &key).is_none());
    }
}
