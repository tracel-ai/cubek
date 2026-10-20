use core::fmt::{Debug, Display};
use cubek_tile::launch::{Refusal, Vectorization};

use crate::Hardware;
use crate::microkernel::{CubePlanes, KernelEstimate};

/// What one owner of a work item runs.
///
/// A value: the choice the hardware settled, sized by its settings, holding the microkernel it is
/// built from where it is built from one. Its scope is in its name.
pub trait Microkernel: Debug + Clone + PartialEq + Sized {
    /// What one owner is handed, as far as the choice of microkernel depends on it.
    type WorkItem: Debug + Clone;
    /// The knobs a measurement settles.
    type Settings: Debug + Clone + Display;
    /// How the family names its operands, for [`Vectorization`].
    type Operand: Debug + Copy + Eq;
    /// How the family places work items over a problem: what a microkernel reads the tiles it is
    /// handed off.
    type Partitioning: Debug + Clone;

    /// Why [`candidates`](Self::candidates) comes back empty on a device, when it does.
    const DECLINE: &'static str;

    /// The settings worth trying on this device, its default first: its own knobs, each setting
    /// holding the one option it picked of the microkernel it is built from. A routine built on it
    /// lists these, each at its own default, before it varies its own knobs; a microkernel built on
    /// it picks from them by name.
    fn settings(hw: &Hardware) -> Vec<Self::Settings>;

    /// Every microkernel this device can run `item` as, best first; empty where it can run none.
    ///
    /// An iterator, and one that allocates nothing: [`best`](Self::best) takes its first item on
    /// every `prepare`, so a list built to be read once would be built on every launch.
    fn candidates(
        hw: &Hardware,
        item: &Self::WorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self>;

    /// Whether this microkernel serves `item` on any device, and why not where it does not: what
    /// [`best`](Self::best) reports before [`DECLINE`](Self::DECLINE), which then names what a
    /// device lacks. Every item, where it can say nothing finer.
    ///
    /// # Errors
    ///
    /// The reason this microkernel serves no such item.
    fn admits(_item: &Self::WorkItem) -> Result<(), Refusal> {
        Ok(())
    }

    /// Why [`candidates`](Self::candidates) comes back empty at `settings`: [`DECLINE`](Self::DECLINE),
    /// where no setting asks the device for something finer.
    fn decline(_settings: &Self::Settings) -> &'static str {
        Self::DECLINE
    }

    /// The best candidate.
    ///
    /// # Errors
    ///
    /// [`admits`](Self::admits)'s reason, then [`decline`](Self::decline)'s where this device runs
    /// none.
    fn best(
        hw: &Hardware,
        item: &Self::WorkItem,
        settings: &Self::Settings,
    ) -> Result<Self, Refusal> {
        Self::admits(item)?;
        Self::candidates(hw, item, settings)
            .next()
            .ok_or(Refusal::Declined {
                why: Self::decline(settings),
            })
    }

    /// Enqueue what the accesses this microkernel's own code makes to the tiles it is handed
    /// require of their widths, under the `partitioning` its routine placed it in, then hand
    /// `vectorization` to the microkernels it is built from. Where those tiles live is its
    /// walker's to know, not its own.
    fn vectorize(
        &self,
        partitioning: &Self::Partitioning,
        vectorization: &mut Vectorization<Self::Operand>,
    );

    /// Set in `estimate` the parts this microkernel owns of a kernel running `item` placed by
    /// `partitioning` on cubes of `planes`, then hand `estimate` to the microkernel it is built from:
    /// the staging where it places the stage's reads, the work where its units hold it.
    fn estimate(
        &self,
        item: &Self::WorkItem,
        partitioning: &Self::Partitioning,
        planes: CubePlanes,
        estimate: &mut KernelEstimate,
    );
}
