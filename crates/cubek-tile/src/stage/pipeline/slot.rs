//! [`Slot`]: a payload plus the [`Meeting`] sequencing its fill against its read.

use cubecl::prelude::*;

use crate::*;

pub(crate) const FIRST_SLOT: usize = 0;

/// One slot of a buffered walk: payload `T` and the rendezvous sequencing fill against read.
#[derive(CubeType)]
pub struct Slot<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Meeting,
    /// When each operand of `T` is filled, in the order `T` holds them.
    #[cube(comptime)]
    pub(crate) refills: Vec<Refill>,
}

#[cube]
impl<T: CubeType> Slot<T> {
    /// Wrap an already-built payload and pipeline.
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

    /// Producer acquire: wait until the slot is free.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn acquire_write(&self) {
        match &self.pipeline {
            Meeting::Barrier { empty, writes, .. } => empty.wait_parity(*writes ^ 1),
            Meeting::Cube => sync_cube(),
        }
    }

    /// Producer release: publish the fill ([`Meeting::producers`] say who arrives).
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
            Meeting::Cube => {}
        }
    }

    /// Consumer acquire: wait for the slot's fill.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn acquire_read(&self) {
        match &self.pipeline {
            Meeting::Barrier { full, reads, .. } => full.wait_parity(*reads),
            Meeting::Cube => {}
        }
    }

    /// Consumer release: free the slot.
    #[allow(dead_code)] // Reached through its expand, from `Slot::fill` / `Slot::consume`.
    pub(crate) fn release_read(&mut self) {
        match &mut self.pipeline {
            Meeting::Barrier { empty, reads, .. } => {
                empty.arrive();
                *reads ^= 1;
            }
            Meeting::Cube => {}
        }
    }

    /// Publish this slot's last fill where no later fill will; call right before
    /// [`consume`](Slot::consume).
    pub fn publish(&self) {
        match &self.pipeline {
            Meeting::Cube => sync_cube(),
            Meeting::Barrier { .. } => {}
        }
    }
}
