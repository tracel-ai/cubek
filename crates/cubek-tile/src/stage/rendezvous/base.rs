//! [`Meeting`], the fill-vs-read rendezvous of one staging slot, and its [`Rendezvous`] strategy.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use crate::*;

/// How a slot rendezvouses its fill against its read, fixed at construction.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rendezvous {
    /// Cooperative copy, synchronized by one `sync_cube` per phase.
    Cube,
    /// Async bulk copy (TMA) over a `full`/`empty` mbarrier pair.
    Barrier,
}

impl Rendezvous {
    /// Join the rendezvous of a slot's sources over a walk `fillers` planes fill; `Barrier` wins.
    pub(crate) fn for_deliveries(deliveries: &[Delivery], fillers: usize) -> Rendezvous {
        assert!(
            !deliveries.is_empty(),
            "Slot: a slot must have at least one delivery"
        );
        let floor = if fillers > 0 {
            Rendezvous::Barrier
        } else {
            Rendezvous::Cube
        };
        deliveries.iter().fold(floor, |sync, delivery| {
            match (sync, delivery.rendezvous()) {
                (Rendezvous::Barrier, _) | (_, Rendezvous::Barrier) => Rendezvous::Barrier,
                (Rendezvous::Cube, Rendezvous::Cube) => Rendezvous::Cube,
            }
        })
    }

    /// Whether a barrier slot needs every unit to publish its writes.
    pub(crate) fn collective_full(deliveries: &[Delivery]) -> bool {
        deliveries
            .iter()
            .any(|delivery| delivery.rendezvous() == Rendezvous::Cube)
    }
}

/// The rendezvous for one slot and the barriers it owns.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum Meeting {
    /// Synchronous cooperative copy, synchronized by one `sync_cube` per phase.
    Cube,
    /// Async producer/consumer over a `full`/`empty` mbarrier pair, one parity each.
    Barrier {
        /// Producer to consumer: flips once declared arrivals and TMA bytes land.
        full: Shared<Barrier>,
        /// Consumer to producer: flips once every reader has freed the slot.
        empty: Shared<Barrier>,
        /// Whether `full` counts every producer's arrival or only the elected issuer's.
        all_publish: bool,
        /// The unit that issues this slot's bulk copies and declares their bytes.
        elected: u32,
        /// `full`'s parity, flipped by the producer's release.
        writes: u32,
        /// `empty`'s parity, flipped by the consumer's release.
        reads: u32,
    },
}

#[cube]
impl Meeting {
    /// Allocate the pipeline for `sync`; for [`Barrier`](Rendezvous::Barrier), arm and fence the
    /// mbarrier pair. Must run while every unit of the cube is present.
    pub(crate) fn new(
        #[comptime] sync: Rendezvous,
        #[comptime] collective_full: bool,
        #[comptime] fillers: usize,
    ) -> Meeting {
        match sync {
            Rendezvous::Cube => Meeting::new_Cube(),
            Rendezvous::Barrier => {
                let full =
                    Barrier::shared(Meeting::producers(collective_full, fillers), UNIT_POS == 0);
                let empty = Barrier::shared(Meeting::consumers(fillers), UNIT_POS == 0);
                sync_async_proxy_shared();
                sync_cube();
                let elected = Meeting::elected(fillers);
                let all_publish = comptime!(collective_full || fillers > 0);
                Meeting::new_Barrier(full, empty, all_publish, elected, 0, 0)
            }
        }
    }

    /// Units that arrive on `full`.
    pub fn producers(#[comptime] collective_full: bool, #[comptime] fillers: usize) -> u32 {
        if comptime!(collective_full) {
            CUBE_DIM
        } else if comptime!(fillers > 0) {
            comptime!(fillers as u32) * CUBE_DIM_X
        } else {
            1u32.runtime()
        }
    }

    /// Units that arrive on `empty`: the ones that read the slot.
    pub fn consumers(#[comptime] fillers: usize) -> u32 {
        if comptime!(fillers == 0) {
            CUBE_DIM
        } else {
            CUBE_DIM - comptime!(fillers as u32) * CUBE_DIM_X
        }
    }

    /// The unit that issues a bulk copy and declares its bytes.
    pub fn elected(#[comptime] fillers: usize) -> u32 {
        if comptime!(fillers == 0) {
            0u32.runtime()
        } else {
            Meeting::consumers(fillers)
        }
    }

    /// Fill staged `dst` from `src`.
    pub fn fill<E: Numeric>(&self, dst: &mut Tile<E>, src: &Tile<E>) {
        // Bound before the match, which borrows the kind.
        let space = comptime!(dst.place.space.clone());
        let decodes = src.scaled();
        if comptime!(decodes) {
            dst.copy_from(src);
        } else {
            self.fill_as_it_lies(dst, src, space);
        }
    }

    /// [`fill`](Meeting::fill) a source that decodes nothing.
    fn fill_as_it_lies<E: Numeric>(
        &self,
        dst: &mut Tile<E>,
        src: &Tile<E>,
        #[comptime] space: Space,
    ) {
        match self {
            Meeting::Barrier { full, elected, .. } => match (&mut dst.kind, &src.kind) {
                (TileKind::Memory(d), TileKind::TmaGmem(s)) => {
                    // Single issuer: the transaction count is the elected unit's alone.
                    if UNIT_POS == *elected {
                        full.expect_tx(d.size_bytes());
                        s.stage_into(d, full);
                    }
                }
                (TileKind::Memory(d), TileKind::Memory(s)) => d.fill_from(s, space),
                (TileKind::Memory(d), TileKind::Procedural(s)) => d.fill_procedural(s, space),
                _ => panic!("Meeting::fill: unsupported kind pairing"),
            },
            Meeting::Cube => dst.copy_from(src),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_and_strided_share_a_cube_pipeline() {
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Procedural, Delivery::Copy], 0),
            Rendezvous::Cube
        );
    }

    #[test]
    fn procedural_and_tma_share_a_barrier_pipeline() {
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Procedural, Delivery::Tma], 0),
            Rendezvous::Barrier
        );
        assert!(Rendezvous::collective_full(&[
            Delivery::Procedural,
            Delivery::Tma
        ]));
    }

    #[test]
    fn pure_tma_keeps_its_single_producer_arrival() {
        assert!(!Rendezvous::collective_full(&[Delivery::Tma]));
    }

    #[test]
    fn a_filled_slot_rendezvouses_on_a_barrier_whatever_delivered_it() {
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Copy], 1),
            Rendezvous::Barrier
        );
    }
}
