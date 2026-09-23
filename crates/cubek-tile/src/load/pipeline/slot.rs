//! The [`Slot`] slot: a matmul-agnostic payload `T` plus the [`Meeting`] sequencing its fill
//! against its read. Generic slot mechanics only: the producer/consumer acquire/release and the
//! final publish; the operand-specific construction and fill live in [`fill`](crate::fill).

use cubecl::prelude::*;
use cubecl::zspace::SmallVec;

use crate::*;

pub(crate) const FIRST_SLOT: usize = 0;

/// Inline capacity for a slot's per-operand plans (spills to heap if exceeded).
pub(crate) const MAX_OPERANDS: usize = 4;

/// Operand positions within a slot's payload, in the order the payload holds them. Named here
/// only because the two stages constructors are one- and two-operand by construction; an
/// operation names its own roles ([`ops::matmul`](crate::ops)).
pub(crate) const FIRST: usize = 0;
pub(crate) const SECOND: usize = 1;

/// When a slot's buffer is brought to its region across the walk. The walk moves each operand's
/// window or it does not, and a window that never moves need be neither refilled nor duplicated
/// per slot; those two savings are the same fact, so one mode carries both.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Refill {
    /// The walk moves this operand's window, so every region refills it.
    EveryRegion,
    /// The window does not move across the walk: filled once, above the loop.
    Once,
    /// Filled once, and this slot reads the first slot's buffer rather than one of its own.
    /// Nobody fills it here.
    Shared,
}

impl Refill {
    /// This operand's refill in a later slot, given its refill in the first: a window that does
    /// not move is read from the first slot's buffer, one that moves is rebuilt per slot.
    pub(crate) fn in_later_slot(self) -> Refill {
        match self {
            Refill::Once => Refill::Shared,
            Refill::EveryRegion => Refill::EveryRegion,
            Refill::Shared => unreachable!("Shared is produced only in later slots"),
        }
    }
}

/// What one operand's stage is within a staging slot: when it is filled, and what moves its
/// bytes into it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub(crate) struct OperandPlan {
    pub mode: Refill,
    pub delivery: Delivery,
}

/// One slot of a buffered walk: its payload `T` and the [`Meeting`] sequencing fill vs read.
/// Generic over `T` and how many operands it holds, so the slot knows nothing of the operation; it
/// hands out a synchronized `&mut T` to fill (`write`) and a synchronized `&T` to consume (`read`).
#[derive(CubeType)]
pub struct Slot<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Meeting,
    /// One entry per operand the payload `T` holds, in the order `T` holds them, so the arity
    /// is the payload's and nothing here has to name a left or a right.
    #[cube(comptime)]
    pub(crate) plans: SmallVec<[OperandPlan; MAX_OPERANDS]>,
}

#[cube]
impl<T: CubeType> Slot<T> {
    /// Wrap an already-built payload and pipeline. (Split out so the tuple `T` never sits in a
    /// struct-literal turbofish, which `#[cube]` can't parse; `Slot::<T>` can.)
    pub(crate) fn wrap(
        data: T,
        pipeline: Meeting,
        #[comptime] plans: SmallVec<[OperandPlan; MAX_OPERANDS]>,
    ) -> Slot<T> {
        Slot::<T> {
            data,
            pipeline,
            plans,
        }
    }

    /// The resolved plan for operand `index`, counted as the payload holds them.
    pub(crate) fn plan(&self, #[comptime] index: usize) -> comptime_type!(OperandPlan) {
        comptime!(*self.plans.get(index).unwrap_or_else(|| panic!(
            "Slot: operand {index} of a slot staging {}",
            self.plans.len()
        )))
    }

    /// Whether this slot has any fixed operand.
    #[allow(dead_code)] // Reached through its expand, from [`StagesFill`].
    pub(crate) fn has_fixed(&self) -> comptime_type!(bool) {
        comptime!(self.plans.iter().any(|p| p.mode == Refill::Once))
    }

    /// Producer acquire: wait the slot is free (`empty`, WAR) for `Barrier`; a `collective` `Cube`
    /// slot rendezvouses on `sync_cube`; a lone-unit one does nothing.
    ///
    /// The first wait is on the parity `writes` was not born at, which a fresh mbarrier already
    /// carries, so it passes straight through.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn acquire_write(&self) {
        match &self.pipeline {
            Meeting::Barrier { empty, writes, .. } => empty.wait_parity(*writes ^ 1),
            Meeting::Cube => sync_cube(),
            Meeting::Solo => {}
        }
    }

    /// Producer release publishes a barrier slot after its required arrivals and any TMA bytes
    /// declared by [`Meeting::fill`] land. Which units arrive is the slot's to say
    /// ([`Meeting::producers`]).
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn release_write(&mut self) {
        match &mut self.pipeline {
            Meeting::Barrier {
                full,
                all_publish,
                elected,
                writes,
                ..
            } => {
                if *all_publish || UNIT_POS == *elected {
                    full.arrive();
                }
                *writes ^= 1;
            }
            Meeting::Cube | Meeting::Solo => {}
        }
    }

    /// Consumer acquire: wait the slot's fill (`full`, RAW) for `Barrier`; nothing for `Cube`
    /// (already rendezvoused in `write`).
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn acquire_read(&self) {
        match &self.pipeline {
            Meeting::Barrier { full, reads, .. } => full.wait_parity(*reads),
            Meeting::Cube | Meeting::Solo => {}
        }
    }

    /// Consumer release: arrive `empty` (free the slot) and flip the read parity for `Barrier`;
    /// nothing for `Cube`.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn release_read(&mut self) {
        match &mut self.pipeline {
            Meeting::Barrier { empty, reads, .. } => {
                empty.arrive();
                *reads ^= 1;
            }
            Meeting::Cube | Meeting::Solo => {}
        }
    }

    /// Publish this slot's last fill when no successor fill's rendezvous will (the walk's final
    /// regions). Only a collective `Cube` slot needs it; callers invoke this immediately before
    /// [`consume`](Slot::consume).
    pub fn publish(&self) {
        match &self.pipeline {
            Meeting::Cube => sync_cube(),
            Meeting::Solo | Meeting::Barrier { .. } => {}
        }
    }
}
