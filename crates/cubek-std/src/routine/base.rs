use core::fmt::{Debug, Display};
use cubecl::tile::launch::Refusal;
use cubecl::tune::Work;

use crate::Hardware;
use crate::microkernel::KernelEstimate;
use crate::routine::Grid;

/// The message a launch expects with when the tile engine refuses a plan its routine prepared:
/// the two disagree, which is a bug in the routine rather than a limit of the device.
pub const ADMITTED_PLAN_RUNS: &str =
    "the selector admitted this plan, so the tile engine launches it and binds its operands";

/// What [`Routine::prepare_first_served`] declines with for a routine listing no
/// setting.
pub const ROUTINE_LISTS_NO_SETTING: Refusal = Refusal::Declined {
    why: "the routine lists no setting on this device",
};

/// How a problem is solved on the hardware it has to run on: the launches it makes, however many
/// that takes, each placing a microkernel, prepared for a device and launched in order.
///
/// What it launches is its own: its blueprint holds every decision of every launch, and its grid
/// what each spawns on this device. A routine of one launch holds one blueprint and spawns one
/// [`Grid`].
pub trait Routine {
    /// The problem, as the host describes it.
    type Problem: Debug + Clone;
    /// The knobs a measurement settles; hold the microkernels'.
    type Settings: Debug + Clone + Display;
    /// Every compile-time decision of its launches: the end result of [`prepare`](Self::prepare).
    type Blueprint: Debug + Clone + PartialEq;
    /// What its launches spawn on this device, beside the blueprint: [`Grid`] for one launch.
    type Grid: Debug + Clone;

    /// The settings worth measuring on this device, the one a caller gets by default first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings>;

    /// The launches for `problem` on this device: the best candidates, placed and vectorized, and
    /// the grids they spawn.
    ///
    /// # Errors
    ///
    /// Why it cannot run here.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &Self::Problem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Self::Grid), Refusal>;

    /// The launches for `problem` at the first of the routine's own [`settings`](Self::settings)
    /// that prepares them on this device: what a caller naming the routine and no setting runs. A
    /// setting whose step does not tile the problem declines, and a shallower one further down
    /// the list often serves it.
    ///
    /// # Errors
    ///
    /// The last setting's decline, where none serves; [`ROUTINE_LISTS_NO_SETTING`] where the
    /// routine lists none.
    fn prepare_first_served(
        &self,
        hw: &Hardware,
        problem: &Self::Problem,
    ) -> Result<(Self::Blueprint, Self::Grid), Refusal> {
        let mut declined = ROUTINE_LISTS_NO_SETTING;
        for settings in self.settings(hw) {
            match self.prepare(hw, problem, &settings) {
                Ok(prepared) => return Ok(prepared),
                Err(why) => declined = why,
            }
        }
        Err(declined)
    }
}

/// A routine of one launch, whose kernel is estimated against its problem.
pub trait EstimatedRoutine: Routine<Problem: EstimatedProblem, Grid = Grid> {
    /// Set in `estimate` the parts this routine owns of the kernel `blueprint` and `grid` describe
    /// for `problem`, then hand `estimate` to its microkernel: the pass walks down as the kernel
    /// does, every component setting its own part.
    fn estimate(
        &self,
        problem: &Self::Problem,
        blueprint: &Self::Blueprint,
        grid: &Grid,
        estimate: &mut KernelEstimate,
    );
}

/// A problem whose kernels are estimated against it: it states its own demand.
pub trait EstimatedProblem {
    /// What the problem asks for, whichever routine runs it.
    fn demand(&self) -> Work;
}
