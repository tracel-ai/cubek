//! One routine at one of its settings, on the device it was listed for ([`ReduceRoutineSetting`]).

use crate::kernels::{
    HypercubeAtomicReduceRoutine, ReduceBlueprint, ReduceKey, ReduceRoutine, ReduceRoutineProblem,
    ReduceTensors, launch_zeroed_atomic, launch_zeroed_atomic_unbound,
};
use core::marker::PhantomData;
use cubecl::prelude::LaunchArg;
use cubecl::tile::Geometry;
use cubecl::tile::launch::{Decline, TileOperand, Unbound};
use cubek_std::Hardware;
use cubek_std::routine::{Grid, Routine};

/// One routine at one of its settings, on the device it was listed for: what a
/// [`ReduceCandidate`](crate::kernels::ReduceCandidate) holds.
#[derive(Debug, Clone)]
pub(crate) struct ReduceRoutineSetting<R: Routine> {
    routine: PhantomData<R>,
    settings: R::Settings,
    /// The routine's prefix and its setting: what a tuning cache names it by.
    name: String,
    hw: Hardware,
}

impl<R, M> ReduceRoutineSetting<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
    R::Settings: core::fmt::Display,
{
    /// Every setting `R` lists on `hw`, each named by the routine's prefix and its setting.
    pub(crate) fn list(hw: &Hardware) -> Vec<Self> {
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

impl<R, M> ReduceRoutineSetting<R>
where
    R: ReduceRoutine<M> + Routine<Microkernel = M>,
{
    /// The routine's prefix and its setting, which a tuning cache is keyed by.
    pub(crate) fn name(&self) -> &String {
        &self.name
    }

    /// `tensors` reduced at this setting.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting.
    pub(crate) fn launch(&self, tensors: &ReduceTensors) -> Result<(), Decline> {
        let (blueprint, grid) = R::prepare(&self.hw, tensors.problem(), &self.settings)?;
        R::launch(tensors, blueprint, grid);
        Ok(())
    }

    /// The launch this setting prepares for the problem `key` describes, or `None` where the
    /// routine declines it on this device.
    pub(crate) fn prepared(&self, key: &ReduceKey) -> Option<(ReduceBlueprint<M>, Grid)> {
        R::prepare(&self.hw, &ReduceRoutineProblem::from(key), &self.settings).ok()
    }

    /// `tensors`, whose input the caller binds, reduced at this setting
    /// ([`ReduceRoutine::launch_unbound`]).
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, its reading the caller's input
    /// included.
    pub(crate) fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        let (blueprint, grid) = R::prepare(&self.hw, tensors.problem(), &self.settings)?;
        R::launch_unbound::<I>(tensors, blueprint, grid, input);
        Ok(())
    }
}

impl ReduceRoutineSetting<HypercubeAtomicReduceRoutine> {
    /// `tensors` summed at this setting into their output, which the zeroing sets first. Both
    /// routines prepare before either launches.
    ///
    /// # Errors
    ///
    /// Why the routine or the zeroing cannot run the problem at its setting.
    pub(crate) fn launch_zeroed(&self, tensors: &ReduceTensors) -> Result<(), Decline> {
        HypercubeAtomicReduceRoutine::prepare(&self.hw, tensors.problem(), &self.settings)
            .and_then(|prepared| launch_zeroed_atomic(&self.hw, tensors, prepared))
    }

    /// [`launch_unbound`](Self::launch_unbound) after the output is zeroed, since the routine only
    /// adds into it.
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, why the zeroing cannot run it, or
    /// why it cannot read the caller's input.
    pub(crate) fn launch_zeroed_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        let prepared =
            HypercubeAtomicReduceRoutine::prepare(&self.hw, tensors.problem(), &self.settings)?;
        launch_zeroed_atomic_unbound::<I>(&self.hw, tensors, prepared, input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::{CubeStagedReduceRoutine, UnitDirectReduceRoutine};
    use cubecl::tile::launch::StoredTileOperand;
    use cubek_microkernel::reduce::ReduceOperation;
    use cubek_std::devices;

    /// The first setting of the cube routine on a device that runs none of it.
    fn declining() -> ReduceRoutineSetting<CubeStagedReduceRoutine> {
        let hw = devices::lacking_every_feature(devices::m2_pro());
        ReduceRoutineSetting::list(&hw).remove(0)
    }

    /// The unit routine's first setting on an M2 Pro.
    fn unit() -> ReduceRoutineSetting<UnitDirectReduceRoutine> {
        ReduceRoutineSetting::list(&devices::m2_pro()).remove(0)
    }

    /// The atomic hypercube's first setting on an M2 Pro.
    fn atomic() -> ReduceRoutineSetting<HypercubeAtomicReduceRoutine> {
        ReduceRoutineSetting::list(&devices::m2_pro()).remove(0)
    }

    /// The key of sums of eight rows of 16 values.
    fn key_of_eight_rows() -> ReduceKey {
        ReduceKey::from(&ReduceRoutineProblem::dense(
            &[8, 16],
            1,
            ReduceOperation::Sum,
        ))
    }

    #[test]
    fn every_setting_of_a_routine_is_listed() {
        let hw = devices::m2_pro();
        let settings: Vec<_> = ReduceRoutineSetting::<UnitDirectReduceRoutine>::list(&hw)
            .into_iter()
            .map(|listed| listed.settings)
            .collect();
        assert_eq!(settings, UnitDirectReduceRoutine::settings(&hw));
    }

    #[test]
    fn a_setting_is_named_by_its_routine_and_setting() {
        let settings = UnitDirectReduceRoutine::settings(&devices::m2_pro())[0];
        assert_eq!(unit().name(), &format!("unit-direct-reduce {settings}"));
    }

    #[test]
    fn a_settings_blueprint_is_the_one_it_prepares_for_the_keys_problem() {
        let key = key_of_eight_rows();
        let hw = devices::m2_pro();
        let (prepared, _) = UnitDirectReduceRoutine::prepare(
            &hw,
            &ReduceRoutineProblem::from(&key),
            &UnitDirectReduceRoutine::settings(&hw)[0],
        )
        .expect("the unit routine serves every problem");
        assert_eq!(
            unit().prepared(&key).map(|(blueprint, _)| blueprint),
            Some(prepared)
        );
    }

    #[test]
    fn a_setting_that_declines_the_key_prepares_nothing() {
        assert!(declining().prepared(&key_of_eight_rows()).is_none());
    }

    #[test]
    fn a_setting_that_declines_the_tensors_does_not_launch() {
        let tensors = ReduceTensors::zeros(&[4, 8], 1, ReduceOperation::Sum);
        assert!(declining().launch(&tensors).is_err());
    }

    #[test]
    fn a_setting_that_declines_an_unbound_input_does_not_ask_for_it() {
        let tensors = ReduceTensors::unbound_zeros(&[4, 8], 1, ReduceOperation::Sum);
        let launched = declining().launch_unbound::<StoredTileOperand>(&tensors, |_| {
            unreachable!("a declined setting binds no input")
        });
        assert!(launched.is_err());
    }

    #[test]
    fn an_atomic_setting_does_not_launch_a_max() {
        let tensors = ReduceTensors::zeros(&[4, 8], 1, ReduceOperation::Max);
        assert!(atomic().launch_zeroed(&tensors).is_err());
    }

    #[test]
    fn an_atomic_setting_does_not_ask_for_a_maxs_unbound_input() {
        let tensors = ReduceTensors::unbound_zeros(&[4, 8], 1, ReduceOperation::Max);
        let launched = atomic().launch_zeroed_unbound::<StoredTileOperand>(&tensors, |_| {
            unreachable!("a declined setting binds no input")
        });
        assert!(launched.is_err());
    }
}
