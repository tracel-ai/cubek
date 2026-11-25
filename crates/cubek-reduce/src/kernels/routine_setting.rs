//! One routine at one of its settings, on the device it was listed for ([`ReduceRoutineSetting`]).

use crate::kernels::{
    HypercubeAtomicReduceRoutine, ReduceBlueprint, ReduceKey, ReduceLauncher, ReduceRoutineProblem,
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
    R: ReduceLauncher<M> + Routine<Microkernel = M>,
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
    R: ReduceLauncher<M> + Routine<Microkernel = M>,
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
    /// ([`ReduceLauncher::launch_unbound`]).
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
