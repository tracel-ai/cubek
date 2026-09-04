//! The [`Staging`] slot: a matmul-agnostic payload `T` plus the [`Pipeline`] sequencing its fill
//! against its read. Generic slot mechanics only: the producer/consumer acquire/release and the
//! final publish; the operand-specific construction and fill live in [`fill`](crate::fill).

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::*;

pub(crate) const FIRST_SLOT: usize = 0;

/// Inline capacity for a slot's per-operand plans (spills to heap if exceeded).
pub(crate) const MAX_OPERANDS: usize = 4;

/// Operand positions within a slot's payload, in the order the payload holds them. Named here
/// only because the two ring constructors are one- and two-operand by construction; an
/// operation names its own roles ([`ops::matmul`](crate::ops)).
pub(crate) const FIRST: usize = 0;
pub(crate) const SECOND: usize = 1;

/// When a slot's buffer is brought to its region across the walk. The walk moves each operand's
/// window or it does not, and a window that never moves need be neither refilled nor duplicated
/// per slot; those two savings are the same fact, so one mode carries both.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum WindowMode {
    /// The walk moves this operand's window, so every region refills it.
    Streamed,
    /// The window is fixed across the walk: filled once, above the loop.
    Fixed,
    /// Fixed, with this slot reusing the first slot's buffer, which is already filled. Nobody
    /// fills it here.
    Reused,
}

impl WindowMode {
    /// This operand's mode in a later ring slot, given its mode in the first: a fixed window
    /// reuses the first slot's buffer, a streamed one is rebuilt per slot.
    pub(crate) fn in_later_slot(self) -> WindowMode {
        match self {
            WindowMode::Fixed => WindowMode::Reused,
            WindowMode::Streamed => WindowMode::Streamed,
            WindowMode::Reused => unreachable!("Reused is produced only in later slots"),
        }
    }
}

/// What one operand's stage is within a staging slot: when it is filled, and what moves its
/// bytes into it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct OperandPlan {
    pub mode: WindowMode,
    pub delivery: Delivery,
}

/// One slot of a buffered walk: its payload `T` and the [`Pipeline`] sequencing fill vs read.
/// Generic over `T` and over how many operands `T` holds, so the slot knows nothing about the
/// operation; it just hands out a synchronized `&mut T` to fill (`write`) and a synchronized
/// `&T` to consume (`read`).
#[derive(CubeType)]
pub struct Staging<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Pipeline,
    /// One entry per operand the payload `T` holds, in the order `T` holds them, so the arity
    /// is the payload's and nothing here has to name a left or a right.
    #[cube(comptime)]
    pub(crate) plans: SmallVec<[OperandPlan; MAX_OPERANDS]>,
}

#[cube]
impl<T: CubeType> Staging<T> {
    /// Wrap an already-built payload and pipeline. (Split out so the tuple `T` never sits in a
    /// struct-literal turbofish, which `#[cube]` can't parse; `Staging::<T>` can.)
    pub(crate) fn wrap(
        data: T,
        pipeline: Pipeline,
        #[comptime] plans: SmallVec<[OperandPlan; MAX_OPERANDS]>,
    ) -> Staging<T> {
        Staging::<T> {
            data,
            pipeline,
            plans,
        }
    }

    /// The resolved plan for operand `index`, counted as the payload holds them.
    pub(crate) fn plan(&self, #[comptime] index: usize) -> comptime_type!(OperandPlan) {
        comptime!(*self.plans.get(index).unwrap_or_else(|| panic!(
            "Staging: operand {index} of a slot staging {}",
            self.plans.len()
        )))
    }

    /// Whether this slot has any fixed operand.
    pub(crate) fn has_fixed(&self) -> comptime_type!(bool) {
        comptime!(self.plans.iter().any(|p| p.mode == WindowMode::Fixed))
    }

    /// Producer acquire: wait the slot is free (`empty`, WAR) for `Barrier`; a `collective` `Cube`
    /// slot rendezvouses on `sync_cube`; a lone-unit one does nothing.
    pub(crate) fn acquire_write(&self) {
        match &self.pipeline {
            Pipeline::Barrier { empty, phase, .. } => empty.wait_parity(*phase ^ 1),
            Pipeline::Cube => sync_cube(),
            Pipeline::Solo => {}
        }
    }

    /// Producer release publishes a barrier slot after its required arrivals and any TMA bytes
    /// declared by [`Pipeline::fill`] land. Mixed slots need every unit; pure TMA uses unit 0.
    pub(crate) fn release_write(&self) {
        match &self.pipeline {
            Pipeline::Barrier {
                full,
                collective_full,
                ..
            } => {
                if comptime!(*collective_full) || UNIT_POS == 0 {
                    full.arrive();
                }
            }
            Pipeline::Cube | Pipeline::Solo => {}
        }
    }

    /// Consumer acquire: wait the slot's fill (`full`, RAW) for `Barrier`; nothing for `Cube` (already
    /// rendezvoused in `write`).
    pub(crate) fn acquire_read(&self) {
        match &self.pipeline {
            Pipeline::Barrier { full, phase, .. } => full.wait_parity(*phase),
            Pipeline::Cube | Pipeline::Solo => {}
        }
    }

    /// Consumer release: arrive `empty` (free the slot) and flip the phase for `Barrier`; nothing for
    /// `Cube`.
    pub(crate) fn release_read(&mut self) {
        match &mut self.pipeline {
            Pipeline::Barrier { empty, phase, .. } => {
                empty.arrive();
                *phase ^= 1;
            }
            Pipeline::Cube | Pipeline::Solo => {}
        }
    }

    /// Publish this slot's last fill when no successor fill's rendezvous will (the walk's final
    /// regions). Only a collective `Cube` slot needs it; callers invoke this immediately before
    /// [`consume`](Staging::consume).
    pub fn publish(&self) {
        match &self.pipeline {
            Pipeline::Cube => sync_cube(),
            Pipeline::Solo | Pipeline::Barrier { .. } => {}
        }
    }
}
