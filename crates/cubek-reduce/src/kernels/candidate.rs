//! A routine at one of its settings, whichever routine it is ([`ReduceCandidate`]), and the launch
//! it prepares for a key ([`ReducePrepared`]).

use crate::kernels::{
    CubeStagedReduceRoutine, HypercubeAtomicReduceRoutine, HypercubeLastArrivalReduceRoutine,
    PlaneShuffleReduceRoutine, ReduceBlueprint, ReduceKey, ReduceRoutineSetting, ReduceTensors,
    UnitDirectReduceRoutine,
};
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
    pub fn launch(&self, tensors: &ReduceTensors) -> Result<(), Decline> {
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
    pub fn prepared(&self, key: &ReduceKey) -> Option<ReducePrepared> {
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
    pub fn name(&self) -> &String {
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
        routine: fn(ReduceBlueprint<M>) -> ReduceCompiled,
        blueprint: ReduceBlueprint<M>,
        grid: Grid,
    ) -> Self {
        Self {
            compiled: routine(blueprint),
            launch: grid.launch,
        }
    }
}

/// Two launches compile alike where their routine and blueprint do: the blueprint's partitioning
/// settles the cube on the device it was listed for, and the count of cubes is a runtime value.
impl PartialEq for ReducePrepared {
    fn eq(&self, other: &Self) -> bool {
        self.compiled == other.compiled
    }
}

/// What a prepared launch compiles: its routine and its blueprint.
#[derive(Clone, PartialEq)]
enum ReduceCompiled {
    UnitDirect(ReduceBlueprint<<UnitDirectReduceRoutine as Routine>::Microkernel>),
    PlaneShuffle(ReduceBlueprint<<PlaneShuffleReduceRoutine as Routine>::Microkernel>),
    CubeStaged(ReduceBlueprint<<CubeStagedReduceRoutine as Routine>::Microkernel>),
    HypercubeLastArrival(
        ReduceBlueprint<<HypercubeLastArrivalReduceRoutine as Routine>::Microkernel>,
    ),
    HypercubeAtomic(ReduceBlueprint<<HypercubeAtomicReduceRoutine as Routine>::Microkernel>),
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::ReduceRoutine;
    use crate::kernels::testing::{
        BufferScopeChecker, Reduction, UnboundScopeChecker, bound_back, sum_of_long_rows,
        unbound_tensors,
    };
    use cubecl::client::Client;
    use cubecl::prelude::TensorBinding;
    use cubecl::tile::launch::StoredTileOperand;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;
    use cubek_test_utils::ScopeChecker;

    const SHAPE: [usize; 2] = [4, 8];

    /// Checks that `setting` made a [`ReduceCandidate`] keeps its name.
    fn keeps_its_name<R: ReduceRoutine<M> + Routine<Microkernel = M>, M>(
        setting: &ReduceRoutineSetting<R>,
    ) where
        ReduceRoutineSetting<R>: Clone,
        ReduceCandidate: From<ReduceRoutineSetting<R>>,
    {
        let named = ReduceCandidate::from(setting.clone()).to_string();
        assert_eq!(&named, setting.name());
    }

    #[test]
    fn a_unit_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineSetting::<UnitDirectReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn a_plane_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineSetting::<PlaneShuffleReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn a_cube_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineSetting::<CubeStagedReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn a_last_arrival_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineSetting::<HypercubeLastArrivalReduceRoutine>::list(&devices::m2_pro())[0],
        );
    }

    #[test]
    fn an_atomic_candidate_keeps_its_name() {
        keeps_its_name(
            &ReduceRoutineSetting::<HypercubeAtomicReduceRoutine>::list(&devices::m2_pro())[0],
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
            ReduceRoutineSetting::<PlaneShuffleReduceRoutine>::list(&lacking_every_feature())
                .remove(0),
        );
    }

    #[test]
    fn a_cube_candidate_declines_an_unbound_input() {
        declines_unbound(
            ReduceRoutineSetting::<CubeStagedReduceRoutine>::list(&lacking_every_feature())
                .remove(0),
        );
    }

    #[test]
    fn a_last_arrival_candidate_declines_an_unbound_input() {
        declines_unbound(
            ReduceRoutineSetting::<HypercubeLastArrivalReduceRoutine>::list(
                &lacking_every_feature(),
            )
            .remove(0),
        );
    }

    #[test]
    fn an_atomic_candidate_declines_an_unbound_input() {
        declines_unbound(
            ReduceRoutineSetting::<HypercubeAtomicReduceRoutine>::list(&lacking_every_feature())
                .remove(0),
        );
    }

    /// Every setting of every routine.
    #[test]
    fn every_setting_of_every_routine_is_listed() {
        let hw = devices::m2_pro();
        let settings = UnitDirectReduceRoutine::settings(&hw).len()
            + PlaneShuffleReduceRoutine::settings(&hw).len()
            + CubeStagedReduceRoutine::settings(&hw).len()
            + HypercubeLastArrivalReduceRoutine::settings(&hw).len()
            + HypercubeAtomicReduceRoutine::settings(&hw).len();
        assert_eq!(ReduceCandidate::list(&hw).len(), settings);
    }

    /// `reduction` launched unbound by `R`'s first candidate on `client`'s device.
    ///
    /// # Errors
    ///
    /// Why the candidate declines it.
    fn launch_unbound_candidate<R, M>(
        client: &Client,
        input: TensorBinding,
        out: TensorBinding,
        reduction: &Reduction,
    ) -> Result<(), Decline>
    where
        R: ReduceRoutine<M> + Routine<Microkernel = M, Settings: core::fmt::Display>,
        ReduceCandidate: From<ReduceRoutineSetting<R>>,
    {
        let hw = Hardware::from_client(client);
        let tensors = unbound_tensors::<R, M>(client, &input, out, reduction)?;
        let candidate = ReduceCandidate::from(ReduceRoutineSetting::<R>::list(&hw).remove(0));
        candidate.launch_unbound::<StoredTileOperand>(&tensors, bound_back(input))
    }

    /// `tensors` reduced by `R`'s first candidate on `hw`.
    ///
    /// # Errors
    ///
    /// Why the candidate declines them.
    fn launch_candidate<R, M>(hw: &Hardware, tensors: &ReduceTensors) -> Result<(), Decline>
    where
        R: ReduceRoutine<M> + Routine<Microkernel = M, Settings: core::fmt::Display>,
        ReduceCandidate: From<ReduceRoutineSetting<R>>,
    {
        ReduceCandidate::from(ReduceRoutineSetting::<R>::list(hw).remove(0)).launch(tensors)
    }

    #[test]
    fn a_unit_candidate_reduces_the_input_its_caller_binds() {
        UnboundScopeChecker::new(launch_unbound_candidate::<UnitDirectReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn a_unit_candidate_reduces_its_buffers() {
        BufferScopeChecker::new(launch_candidate::<UnitDirectReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn a_plane_candidate_reduces_the_input_its_caller_binds() {
        UnboundScopeChecker::new(launch_unbound_candidate::<PlaneShuffleReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn a_plane_candidate_reduces_its_buffers() {
        BufferScopeChecker::new(launch_candidate::<PlaneShuffleReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn a_cube_candidate_reduces_the_input_its_caller_binds() {
        UnboundScopeChecker::new(launch_unbound_candidate::<CubeStagedReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn a_cube_candidate_reduces_its_buffers() {
        BufferScopeChecker::new(launch_candidate::<CubeStagedReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn a_last_arrival_candidate_reduces_the_input_its_caller_binds() {
        UnboundScopeChecker::new(launch_unbound_candidate::<HypercubeLastArrivalReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn a_last_arrival_candidate_reduces_its_buffers() {
        BufferScopeChecker::new(launch_candidate::<HypercubeLastArrivalReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn an_atomic_candidate_reduces_the_input_its_caller_binds() {
        UnboundScopeChecker::new(launch_unbound_candidate::<HypercubeAtomicReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }

    #[test]
    fn an_atomic_candidate_reduces_its_buffers() {
        BufferScopeChecker::new(launch_candidate::<HypercubeAtomicReduceRoutine, _>)
            .check([sum_of_long_rows()]);
    }
}
