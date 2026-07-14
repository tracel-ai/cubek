//! The [`Staging`] slot: a matmul-agnostic payload `T` plus the [`Pipeline`] sequencing its fill
//! against its read. Generic slot mechanics only — the producer/consumer acquire/release and the
//! final publish; the operand-specific construction and fill live in [`fill`](crate::fill).

use cubecl::prelude::*;

use crate::*;

/// One slot of the staged `mma` pipeline: its payload `T` and the [`Pipeline`] sequencing fill vs
/// read. Generic over `T`, so the slot is matmul-agnostic — it just hands out a synchronized `&mut T`
/// to fill (`write`) and a synchronized `&T` to consume (`read`).
#[derive(CubeType)]
pub struct Staging<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Pipeline,
    /// Operands the walk leaves invariant: filled once by [`fill_pinned`](Staging::fill_pinned),
    /// skipped by [`fill_streamed`](Staging::fill_streamed). Only the `(Tile, Tile)` payload sets
    /// these; a generic slot pins nothing.
    #[cube(comptime)]
    pub(crate) pin_lhs: bool,
    #[cube(comptime)]
    pub(crate) pin_rhs: bool,
}

#[cube]
impl<T: CubeType> Staging<T> {
    /// Wrap an already-built payload and pipeline. Private: the public entry is the operand-deducing
    /// [`new`](Staging::new). (Split out so the tuple `T` never sits in a struct-literal turbofish,
    /// which `#[cube]` can't parse; `Staging::<T>` can.)
    pub(crate) fn wrap(
        data: T,
        pipeline: Pipeline,
        #[comptime] pin_lhs: bool,
        #[comptime] pin_rhs: bool,
    ) -> Staging<T> {
        Staging::<T> {
            data,
            pipeline,
            pin_lhs,
            pin_rhs,
        }
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

    /// Producer release: arrive `full` (elected unit) so the consumer's `full` wait can pass. The
    /// bytes were declared per tile by [`Pipeline::fill`], so this is a bare arrival. No-op for `Cube`.
    pub(crate) fn release_write(&self) {
        match &self.pipeline {
            Pipeline::Barrier { full, .. } => {
                if UNIT_POS == 0 {
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
    /// regions). Only a collective `Cube` slot needs it; reached only through
    /// [`consume_final`](Staging::consume_final).
    pub(crate) fn publish(&self) {
        match &self.pipeline {
            Pipeline::Cube => sync_cube(),
            Pipeline::Solo | Pipeline::Barrier { .. } => {}
        }
    }
}
