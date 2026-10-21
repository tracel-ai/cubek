use core::fmt::{Debug, Display};
use cubecl::tune::Work;
use cubecl::tile::launch::{Refusal, Vectorization};

use crate::Hardware;
use crate::microkernel::{KernelEstimate, Microkernel};
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

/// A whole kernel written around one microkernel.
pub trait Routine {
    /// The problem, as the host describes it.
    type Problem: Debug + Clone + EstimatedProblem;
    /// The microkernel whose work items this routine tiles the problem into.
    type Microkernel: Microkernel;
    /// The knobs a measurement settles; hold the microkernel's.
    type Settings: Debug + Clone + Display;
    /// Every compile-time decision of a launch: the end result of [`prepare`](Self::prepare).
    type Blueprint: Debug + Clone + PartialEq;
    /// How it places its microkernel's work items over a problem, which its own code's accesses
    /// are read off.
    type Partitioning: Debug + Clone;

    /// The settings worth measuring on this device, the one a caller gets by default first.
    fn settings(&self, hw: &Hardware) -> Vec<Self::Settings>;

    /// Enqueue what the accesses this routine's own code makes under `partitioning` require of
    /// the operands' widths, then hand `vectorization` to `microkernel` with what it is placed
    /// in: the pass walks down as the kernel does.
    fn vectorize(
        &self,
        microkernel: &Self::Microkernel,
        partitioning: &Self::Partitioning,
        vectorization: &mut Vectorization<<Self::Microkernel as Microkernel>::Operand>,
    );

    /// The launch for `problem` on this device: the best candidate, placed and vectorized, and the
    /// grid it spawns.
    ///
    /// # Errors
    ///
    /// Why it cannot run here.
    fn prepare(
        &self,
        hw: &Hardware,
        problem: &Self::Problem,
        settings: &Self::Settings,
    ) -> Result<(Self::Blueprint, Grid), Refusal>;

    /// The launch for `problem` at the first of the routine's own [`settings`](Self::settings)
    /// that prepares one on this device: what a caller naming the routine and no setting runs. A
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
    ) -> Result<(Self::Blueprint, Grid), Refusal> {
        let mut declined = ROUTINE_LISTS_NO_SETTING;
        for settings in self.settings(hw) {
            match self.prepare(hw, problem, &settings) {
                Ok(prepared) => return Ok(prepared),
                Err(why) => declined = why,
            }
        }
        Err(declined)
    }

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
