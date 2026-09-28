//! The [`Meeting`]: the fill-vs-read rendezvous for one staging slot, and the [`Rendezvous`] strategy
//! deduced from the operands' delivery. [`Barrier`](Rendezvous::Barrier) mirrors cubek-matmul's
//! `specialized/matmul.rs`; [`Cube`](Rendezvous::Cube) and [`Solo`](Rendezvous::Solo) are degenerate cases.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use crate::*;

/// How a slot rendezvouses its fill against its read; fixed comptime at construction
/// from the operands' delivery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rendezvous {
    /// One unit fills and reads its own slot: no collective (single-plane / CPU).
    Solo,
    /// Cooperative element copy rendezvoused on one cube-wide `sync_cube` per phase. The sync sits
    /// in `write` and covers both this slot's fill→read and the sibling's read→refill.
    Cube,
    /// Hardware async bulk copy (TMA): `full`/`empty` mbarrier pair with a `phase` parity, producer
    /// and consumer decoupled so the copy overlaps compute.
    Barrier,
}

impl Rendezvous {
    /// Join the rendezvous requirements of a slot's sources, over a walk `fillers` planes fill.
    /// `Barrier` dominates `Cube` because TMA transaction completion must be in the slot's
    /// publication, while `Cube` rendezvouses on `sync_cube`, where the two roles never meet.
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
                (Rendezvous::Solo, _) | (_, Rendezvous::Solo) => {
                    unreachable!("source rendezvous is never Solo")
                }
            }
        })
    }

    /// Whether a barrier slot needs every unit to publish its writes. Pure TMA has one hardware
    /// issuer; a mixed slot also contains a synchronous cooperative fill.
    pub(crate) fn collective_full(deliveries: &[Delivery]) -> bool {
        deliveries
            .iter()
            .any(|delivery| delivery.rendezvous() == Rendezvous::Cube)
    }
}

/// The rendezvous for one slot, and every barrier it owns. The acquire/release operations live
/// on [`Slot`]; [`fill`](Meeting::fill) is the one op a `write` body reaches for directly.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum Meeting {
    /// Synchronous cooperative element copy, rendezvoused on one `sync_cube` per phase.
    /// The variant (not a flag) carries the choice, so the dispatch is comptime and the
    /// rendezvous emits a bare barrier, never a branch-wrapped one.
    Cube,
    /// A single unit fills and reads its own slot: no collective at all.
    Solo,
    /// Async producer/consumer decoupled over a `full`/`empty` mbarrier pair, one parity each,
    /// so the fill overlaps compute. TMA motivates it, but the barrier itself is
    /// delivery-agnostic; see [`Meeting::fill`].
    ///
    /// Every field here is a runtime value, including the two the construction settles once:
    /// the `CubeType` derive gives an enum variant's fields no comptime spelling. They are
    /// constants the backend folds, not decisions this code can branch on at comptime.
    Barrier {
        /// Producer→consumer: flips after its declared arrivals and all declared TMA transaction
        /// bytes land.
        full: Shared<Barrier>,
        /// Consumer→producer (one arrival per unit that reads): flips once every one of them has
        /// read and freed the slot.
        empty: Shared<Barrier>,
        /// Whether `full` counts every producer's arrival or only the elected issuer's
        /// ([`Meeting::producers`]).
        all_publish: bool,
        /// The one unit that issues this slot's bulk copies and declares their bytes
        /// ([`Meeting::elected`]).
        elected: u32,
        /// `full`'s parity, flipped by the producer's release. Two parities, not one: a unit
        /// that only fills never reaches a read, so a single counter would stall at the first
        /// slot it filled.
        writes: u32,
        /// `empty`'s parity, flipped by the consumer's release.
        reads: u32,
    },
}

#[cube]
impl Meeting {
    /// Allocate the pipeline for `sync`: the `full`/`empty` mbarrier pair, sealed by a proxy fence
    /// before any bulk copy, for [`Barrier`](Rendezvous::Barrier); nothing to allocate otherwise.
    ///
    /// Both barriers are armed and fenced before any plane takes a role, which is why the
    /// election here is unit 0 and the `sync_cube` is the whole cube's: every unit is still
    /// present.
    pub(crate) fn new(
        #[comptime] sync: Rendezvous,
        #[comptime] collective_full: bool,
        #[comptime] fillers: usize,
    ) -> Meeting {
        match sync {
            Rendezvous::Solo => Meeting::new_Solo(),
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

    /// Units that arrive on `full`: the one unit that issued a bulk copy into a slot no plane was
    /// set aside for; every unit that wrote a cooperative fill; every unit of planes filling on
    /// their own, since nothing else keeps them in step and a plane a lap behind waits forever.
    ///
    /// The elected unit is one of the arrivals, and it declares the transaction bytes before it
    /// arrives, so the phase cannot complete on the others' arrivals with the bytes still to
    /// come.
    pub fn producers(#[comptime] collective_full: bool, #[comptime] fillers: usize) -> u32 {
        if comptime!(collective_full) {
            CUBE_DIM
        } else if comptime!(fillers > 0) {
            comptime!(fillers as u32) * CUBE_DIM_X
        } else {
            1u32.runtime()
        }
    }

    /// Units that arrive on `empty`: the ones that read the slot. The planes that only fill sit
    /// at the end of the cube, so what is left below them computes; `CUBE_DIM_X` is a plane's
    /// width, which is how the partitioning laid the cube out ([`Partitioning::cube_dim`]).
    pub fn consumers(#[comptime] fillers: usize) -> u32 {
        if comptime!(fillers == 0) {
            CUBE_DIM
        } else {
            CUBE_DIM - comptime!(fillers as u32) * CUBE_DIM_X
        }
    }

    /// The one unit that issues a bulk copy and declares its bytes: the first producer. That is
    /// the first filling plane's first unit, or unit 0 where no plane was set aside and every
    /// unit produces.
    pub fn elected(#[comptime] fillers: usize) -> u32 {
        if comptime!(fillers == 0) {
            0u32.runtime()
        } else {
            Meeting::consumers(fillers)
        }
    }

    /// Fill staged `dst` from `src`, the one operation a `fill` body performs. A `Barrier` slot
    /// stages under `full`, a `Cube` slot is a blocking [`copy_from`](Tile::copy_from);
    /// in-place operands allocate no destination and never reach it, reading the source instead.
    pub fn fill<E: Numeric>(&self, dst: &mut Tile<E>, src: &Tile<E>) {
        // Bound before the match, which borrows the kind: the fill needs the logical space both
        // sides carry (a gathered source is addressed per axis).
        let space = comptime!(dst.place.space.clone());
        match self {
            Meeting::Barrier { full, elected, .. } => match (&mut dst.kind, &src.kind) {
                (TileKind::Memory(d), TileKind::TmaGmem(s)) => {
                    // One issuer, and the same unit that declares the bytes: the transaction
                    // count is that unit's alone, so a second issuer would over-count the stage.
                    if UNIT_POS == *elected {
                        full.expect_tx(d.size_bytes());
                        s.stage_into(d, full);
                    }
                }
                // A strided source under a barrier is a plain synchronous copy.
                (TileKind::Memory(d), TileKind::Memory(s)) => d.fill_from(s, space),
                (TileKind::Memory(d), TileKind::Procedural(s)) => d.fill_procedural(s, space),
                _ => panic!("Meeting::fill: unsupported kind pairing"),
            },
            Meeting::Cube | Meeting::Solo => dst.copy_from(src),
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

    /// `sync_cube` needs every unit of the cube, and a walk that sets planes aside to fill has
    /// none of its slots reached by all of them.
    #[test]
    fn a_filled_slot_rendezvouses_on_a_barrier_whatever_delivered_it() {
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Copy], 1),
            Rendezvous::Barrier
        );
    }
}
