//! One routine at one of its settings, as the autotune races it ([`ReduceRoutineRunner`]).

use crate::kernels::{
    HypercubeAtomicReduceRoutine, ReduceBlueprint, ReduceRoutine, ReduceRoutineProblem,
    ReduceTensors, launch_zeroed_atomic, launch_zeroed_atomic_unbound, tune::*,
};
use core::marker::PhantomData;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Decline, TileOperand, Unbound};
use cubek_std::Hardware;
use cubek_std::autotune::{CandidatePriority, TuneCandidate, short_of_the_device};
use cubek_std::routine::{Grid, Routine};

/// Runs one routine at one of its settings, on the device the autotune is built for: one
/// candidate of the autotune.
#[derive(Debug, Clone)]
pub(crate) struct ReduceRoutineRunner<R: Routine> {
    routine: PhantomData<R>,
    settings: R::Settings,
    /// The routine's prefix and its setting: what the tuning cache names the candidate by.
    name: String,
    hw: Hardware,
}

impl<R, M> ReduceRoutineRunner<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
    R::Settings: core::fmt::Display,
{
    /// Every setting `R` lists on `hw`, as candidates, each named by the routine's prefix
    /// and its setting.
    pub fn list(hw: &Hardware) -> Vec<Self> {
        R::settings(hw)
            .into_iter()
            .map(|settings| Self {
                routine: PhantomData,
                name: format!("{} {settings}", R::NAME),
                settings,
                hw: hw.clone(),
            })
            .collect()
    }
}

impl<R, M> ReduceRoutineRunner<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
{
    /// `tensors`, whose input the caller binds, reduced at this candidate's setting
    /// ([`ReduceRoutine::launch_unbound`]).
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, its reading the caller's input
    /// included.
    pub fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        let (blueprint, grid) = R::prepare(&self.hw, tensors.problem(), &self.settings)?;
        R::launch_unbound::<I>(tensors, blueprint, grid, input)
    }
}

impl<R, M> ReduceRoutineRunner<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
{
    /// `tensors` reduced at this candidate's setting.
    pub(super) fn run(&self, tensors: ReduceTensors) -> Result<(), String> {
        R::prepare(&self.hw, tensors.problem(), &self.settings)
            .and_then(|(blueprint, grid)| R::launch(&tensors, blueprint, grid))
            .map_err(|decline| decline.to_string())
    }

    /// The launch this candidate prepares for the problem `key` describes, or `None` where the
    /// routine declines it on this device.
    pub(super) fn prepared(&self, key: &ReduceKey) -> Option<(ReduceBlueprint<M>, Grid)> {
        R::prepare(&self.hw, &ReduceRoutineProblem::from(key), &self.settings).ok()
    }
}

impl ReduceRoutineRunner<HypercubeAtomicReduceRoutine> {
    /// `tensors` summed at this candidate's setting into their output, which the zeroing, timed
    /// with it, sets first. Both routines prepare before either launches.
    pub(super) fn run_zeroed(&self, tensors: ReduceTensors) -> Result<(), String> {
        HypercubeAtomicReduceRoutine::prepare(&self.hw, tensors.problem(), &self.settings)
            .and_then(|prepared| launch_zeroed_atomic(&self.hw, &tensors, prepared))
            .map_err(|decline| decline.to_string())
    }

    /// [`launch_unbound`](Self::launch_unbound) after the output is zeroed, since the routine only
    /// adds into it.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, why the zeroing cannot run it, or
    /// why it cannot read the caller's input.
    pub(super) fn launch_zeroed_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        let prepared =
            HypercubeAtomicReduceRoutine::prepare(&self.hw, tensors.problem(), &self.settings)?;
        launch_zeroed_atomic_unbound::<I>(&self.hw, tensors, prepared, input)
    }
}

impl<R, M: PartialEq> TuneCandidate for ReduceRoutineRunner<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
{
    type Strategy = String;
    type Key = ReduceKey;
    type Blueprint = ReduceBlueprint<M>;

    fn strategy(&self) -> &String {
        &self.name
    }

    /// First, lowered a level for each halving the prepared grid falls short of filling the
    /// device; last where the routine declines the problem.
    fn priority(&self, key: &ReduceKey) -> CandidatePriority {
        match self.prepared(key) {
            Some((_, grid)) => {
                CandidatePriority::FIRST.lowered(short_of_the_device(&self.hw, &grid.launch))
            }
            None => CandidatePriority::LAST,
        }
    }

    fn blueprint(&self, key: &ReduceKey) -> Option<Self::Blueprint> {
        self.prepared(key).map(|(blueprint, _)| blueprint)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{CubeStagedReduceRoutine, UnitDirectReduceRoutine};
    use cubecl::tile::launch::StoredTileOperand;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// The first candidate of the cube routine on a device that runs none of it.
    fn declining() -> ReduceRoutineRunner<CubeStagedReduceRoutine> {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        ReduceRoutineRunner::list(&hw).remove(0)
    }

    /// The unit routine's first candidate on an M2 Pro.
    fn unit() -> ReduceRoutineRunner<UnitDirectReduceRoutine> {
        ReduceRoutineRunner::list(&devices::m2_pro()).remove(0)
    }

    /// The atomic hypercube's first candidate on an M2 Pro.
    fn atomic() -> ReduceRoutineRunner<HypercubeAtomicReduceRoutine> {
        ReduceRoutineRunner::list(&devices::m2_pro()).remove(0)
    }

    const ROW_LEN: usize = 16;

    /// The key of sums of `rows` rows of [`ROW_LEN`] values.
    fn key_of_rows(rows: usize) -> ReduceKey {
        ReduceKey::from(&ReduceRoutineProblem::dense(
            &[rows, ROW_LEN],
            1,
            ReduceOperation::Sum,
        ))
    }

    #[test]
    fn every_setting_of_a_routine_is_a_candidate() {
        let hw = devices::m2_pro();
        let settings: Vec<_> = ReduceRoutineRunner::<UnitDirectReduceRoutine>::list(&hw)
            .into_iter()
            .map(|candidate| candidate.settings)
            .collect();
        assert_eq!(settings, UnitDirectReduceRoutine::settings(&hw));
    }

    #[test]
    fn a_candidate_is_named_by_its_routine_and_setting() {
        let settings = UnitDirectReduceRoutine::settings(&devices::m2_pro())[0];
        assert_eq!(unit().strategy(), &format!("unit-direct-reduce {settings}"));
    }

    #[test]
    fn a_candidate_that_fills_the_device_comes_first() {
        assert_eq!(
            unit().priority(&key_of_rows(1 << 18)),
            CandidatePriority::FIRST
        );
    }

    /// Eight rows take one cube of 128 units, where an M2 Pro takes 76.
    #[test]
    fn a_candidate_that_leaves_the_device_idle_is_lowered_a_level_a_halving() {
        assert_eq!(
            unit().priority(&key_of_rows(8)),
            CandidatePriority::FIRST.lowered(76u32.ilog2())
        );
    }

    #[test]
    fn a_candidate_that_declines_the_key_comes_last() {
        assert_eq!(
            declining().priority(&key_of_rows(8)),
            CandidatePriority::LAST
        );
    }

    #[test]
    fn a_candidates_blueprint_is_the_one_it_prepares_for_the_keys_problem() {
        let key = key_of_rows(8);
        let hw = devices::m2_pro();
        let (prepared, _) = UnitDirectReduceRoutine::prepare(
            &hw,
            &ReduceRoutineProblem::from(&key),
            &UnitDirectReduceRoutine::settings(&hw)[0],
        )
        .expect("the unit routine serves every problem");
        assert_eq!(unit().blueprint(&key), Some(prepared));
    }

    #[test]
    fn a_candidate_that_declines_the_key_has_no_blueprint() {
        assert_eq!(declining().blueprint(&key_of_rows(8)), None);
    }

    #[test]
    fn a_candidate_that_declines_the_tensors_does_not_run() {
        let tensors = ReduceTensors::zeros(&[4, 8], 1, ReduceOperation::Sum);
        assert!(declining().run(tensors).is_err());
    }

    #[test]
    fn a_candidate_that_declines_an_unbound_input_does_not_ask_for_it() {
        let tensors = ReduceTensors::unbound_zeros(&[4, 8], 1, ReduceOperation::Sum);
        let launched = declining().launch_unbound::<StoredTileOperand>(&tensors, |_| {
            unreachable!("a declined candidate binds no input")
        });
        assert!(launched.is_err());
    }

    #[test]
    fn an_atomic_candidate_does_not_run_a_max() {
        let tensors = ReduceTensors::zeros(&[4, 8], 1, ReduceOperation::Max);
        assert!(atomic().run_zeroed(tensors).is_err());
    }

    #[test]
    fn an_atomic_candidate_does_not_ask_for_a_maxs_unbound_input() {
        let tensors = ReduceTensors::unbound_zeros(&[4, 8], 1, ReduceOperation::Max);
        let launched = atomic().launch_zeroed_unbound::<StoredTileOperand>(&tensors, |_| {
            unreachable!("a declined candidate binds no input")
        });
        assert!(launched.is_err());
    }
}
