//! The [`Slot`] slot: a [`Payload`](crate::load::pipeline::payload::Payload) plus the [`Meeting`]
//! sequencing its fill against its read. Generic slot mechanics only: the producer/consumer
//! acquire/release and the final publish; what a payload is made of and how it is brought to a
//! region are the payload's own.

use cubecl::prelude::*;

use crate::*;

pub(crate) const FIRST_SLOT: usize = 0;

/// One slot of a buffered walk: its payload `T` and the [`Meeting`] sequencing fill vs read.
/// Generic over `T` and how many operands it holds, so the slot knows nothing of the operation; it
/// hands out a synchronized `&mut T` to fill (`write`) and a synchronized `&T` to consume (`read`).
#[derive(CubeType)]
pub struct Slot<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Meeting,
    /// When each operand the payload `T` holds is filled, in the order `T` holds them, so the
    /// arity is the payload's and nothing here has to name a left or a right.
    #[cube(comptime)]
    pub(crate) refills: Vec<Refill>,
}

#[cube]
impl<T: CubeType> Slot<T> {
    /// Wrap an already-built payload and pipeline. (Split out so the tuple `T` never sits in a
    /// struct-literal turbofish, which `#[cube]` can't parse; `Slot::<T>` can.)
    pub(crate) fn wrap(data: T, pipeline: Meeting, #[comptime] refills: Vec<Refill>) -> Slot<T> {
        Slot::<T> {
            data,
            pipeline,
            refills,
        }
    }

    /// Whether this slot has any fixed operand.
    #[allow(dead_code)] // Reached through its expand, from [`StagesFill`].
    pub(crate) fn has_fixed(&self) -> comptime_type!(bool) {
        comptime!(self.refills.contains(&Refill::Once))
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
