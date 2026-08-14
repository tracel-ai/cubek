//! The [`Staging`] slot: a matmul-agnostic payload `T` plus the [`Pipeline`] sequencing its fill
//! against its read. Generic slot mechanics only: the producer/consumer acquire/release and the
//! final publish; the operand-specific construction and fill live in [`fill`](crate::fill).

use core::option::Option;
use cubecl::prelude::*;

use crate::*;

/// When one operand of a slot is filled across the walk. The walk moves each operand's window or
/// it does not, and a window that never moves need be neither refilled nor duplicated per slot;
/// those two savings are the same fact, so one state carries both.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Fill {
    /// The walk moves this operand's window, so every region refills it.
    Streamed,
    /// The window is fixed across the walk: filled once, above the loop.
    Pinned,
    /// Fixed, and this slot's buffer is the ring's first slot's, which already filled it. Nobody
    /// fills it here. Only a materialized ([`Smem`](Residence::Smem)) operand reaches this: an
    /// in-place payload allocates nothing, so there is no buffer to share and every slot rebinds
    /// its own.
    Shared,
}

impl Fill {
    /// Whether [`fill_pinned`](Staging::fill_pinned) fills this operand: it owns a fixed window.
    pub(crate) fn is_pinned(self) -> bool {
        matches!(self, Fill::Pinned)
    }

    /// Whether [`fill_streamed`](Staging::fill_streamed) fills this operand.
    pub(crate) fn is_streamed(self) -> bool {
        matches!(self, Fill::Streamed)
    }

    /// This operand's state in a later ring slot, given its state in the first. A pinned
    /// materialized operand hands its buffer over; anything else is rebuilt per slot.
    pub(crate) fn shared_by(self, residence: Residence) -> Fill {
        match self {
            Fill::Pinned => match residence {
                Residence::Smem => Fill::Shared,
                Residence::InPlace | Residence::Plane => Fill::Pinned,
                Residence::Auto => panic!("Residence::Auto must be resolved before staging"),
            },
            Fill::Streamed => match residence {
                Residence::InPlace | Residence::Smem | Residence::Plane => Fill::Streamed,
                Residence::Auto => panic!("Residence::Auto must be resolved before staging"),
            },
            Fill::Shared => match residence {
                Residence::InPlace | Residence::Smem | Residence::Plane => Fill::Shared,
                Residence::Auto => panic!("Residence::Auto must be resolved before staging"),
            },
        }
    }
}

/// One slot of the staged `mma` pipeline: its payload `T` and the [`Pipeline`] sequencing fill vs
/// read. Generic over `T`, so the slot is matmul-agnostic; it just hands out a synchronized `&mut T`
/// to fill (`write`) and a synchronized `&T` to consume (`read`).
#[derive(CubeType)]
pub struct Staging<T: CubeType> {
    pub(crate) data: T,
    pub(crate) pipeline: Pipeline,
    /// When each operand is filled, and where it lives at this level; both resolved when the slot
    /// was built. The slot's payload `T` fixes its arity, so a unary slot's right-hand entries are
    /// `None` and asking for them is a bug, not a default.
    #[cube(comptime)]
    pub(crate) fill_lhs: Fill,
    #[cube(comptime)]
    pub(crate) fill_rhs: Option<Fill>,
    #[cube(comptime)]
    pub(crate) residence_lhs: Residence,
    #[cube(comptime)]
    pub(crate) residence_rhs: Option<Residence>,
}

#[cube]
impl<T: CubeType> Staging<T> {
    /// Wrap an already-built payload and pipeline. Private: the public entry is the operand-deducing
    /// [`new`](Staging::new). (Split out so the tuple `T` never sits in a struct-literal turbofish,
    /// which `#[cube]` can't parse; `Staging::<T>` can.)
    pub(crate) fn wrap(
        data: T,
        pipeline: Pipeline,
        #[comptime] fill_lhs: Fill,
        #[comptime] fill_rhs: Option<Fill>,
        #[comptime] residence_lhs: Residence,
        #[comptime] residence_rhs: Option<Residence>,
    ) -> Staging<T> {
        Staging::<T> {
            data,
            pipeline,
            fill_lhs,
            fill_rhs,
            residence_lhs,
            residence_rhs,
        }
    }

    /// Whether this slot's operand (or lhs) is pinned.
    pub(crate) fn is_pinned(&self) -> comptime_type!(bool) {
        comptime!(self.fill_lhs.is_pinned())
    }

    /// Whether this slot's operand (or lhs) is streamed.
    pub(crate) fn is_streamed(&self) -> comptime_type!(bool) {
        comptime!(self.fill_lhs.is_streamed())
    }

    /// When this slot's right-hand operand is filled.
    fn fill_rhs(&self) -> comptime_type!(Fill) {
        comptime!(self.fill_rhs.expect("Staging: unary slot has no rhs fill"))
    }

    /// Whether this slot's rhs operand is pinned.
    pub(crate) fn is_pinned_rhs(&self) -> comptime_type!(bool) {
        let fill = self.fill_rhs();
        comptime!(fill.is_pinned())
    }

    /// Whether this slot's rhs operand is streamed.
    pub(crate) fn is_streamed_rhs(&self) -> comptime_type!(bool) {
        let fill = self.fill_rhs();
        comptime!(fill.is_streamed())
    }

    /// Where this slot's operand (or left-hand operand) lives.
    pub(crate) fn residence(&self) -> comptime_type!(Residence) {
        comptime!(self.residence_lhs)
    }

    /// Where this slot's right-hand operand lives.
    pub(crate) fn residence_rhs(&self) -> comptime_type!(Residence) {
        comptime!(
            self.residence_rhs
                .expect("Staging: binary slot has no rhs residence")
        )
    }

    /// Whether either operand uses a plane partition, requiring an unrolled walk.
    pub(crate) fn has_plane_stage(&self) -> comptime_type!(bool) {
        comptime!(
            self.residence_lhs == Residence::Plane
                || matches!(self.residence_rhs, Option::Some(Residence::Plane))
        )
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
    /// regions). Only a collective `Cube` slot needs it; reached only through
    /// [`consume_final`](Staging::consume_final).
    pub(crate) fn publish(&self) {
        match &self.pipeline {
            Pipeline::Cube => sync_cube(),
            Pipeline::Solo | Pipeline::Barrier { .. } => {}
        }
    }
}
