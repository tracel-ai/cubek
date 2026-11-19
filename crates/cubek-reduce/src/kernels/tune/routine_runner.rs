//! One routine at one of its settings, as the autotune races it ([`ReduceRoutineRunner`]).

use crate::kernels::{
    HypercubeAtomicReduceRoutine, ReduceBlueprint, ReduceLauncher, ReduceRoutineProblem,
    ReduceTensors, launch_zeroed_atomic, launch_zeroed_atomic_unbound, tune::*,
};
use core::marker::PhantomData;
use cubecl::CubeDim;
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
    R: ReduceLauncher<M> + Routine<Microkernel = M>,
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
    R: ReduceLauncher<M> + Routine<Microkernel = M>,
{
    /// `tensors` reduced at this candidate's setting.
    pub(super) fn run(&self, tensors: ReduceTensors) -> Result<(), String> {
        let (blueprint, grid) = R::prepare(&self.hw, tensors.problem(), &self.settings)
            .map_err(|decline| decline.to_string())?;
        R::launch(&tensors, blueprint, grid);
        Ok(())
    }

    /// `tensors`, whose input the caller binds, reduced at this candidate's setting
    /// ([`ReduceLauncher::launch_unbound`]).
    ///
    /// # Errors
    ///
    /// Why the routine cannot run the problem at its setting, or why it cannot read the caller's
    /// input.
    pub fn launch_unbound<I: TileOperand>(
        &self,
        tensors: &ReduceTensors<Geometry>,
        input: impl FnOnce(Unbound) -> <I::Arg as LaunchArg>::RuntimeArg,
    ) -> Result<(), Decline> {
        let (blueprint, grid) = R::prepare(&self.hw, tensors.problem(), &self.settings)?;
        R::launch_unbound::<I>(tensors, blueprint, grid, input)
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
    R: ReduceLauncher<M> + Routine<Microkernel = M>,
{
    type Strategy = String;
    type Key = ReduceKey;
    /// The blueprint and the cube: two settings preparing one blueprint can still launch
    /// different cubes.
    type Blueprint = (ReduceBlueprint<M>, CubeDim);

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
        self.prepared(key)
            .map(|(blueprint, grid)| (blueprint, grid.launch.cube_dim))
    }
}
