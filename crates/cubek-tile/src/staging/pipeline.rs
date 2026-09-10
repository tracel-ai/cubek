//! The [`Pipeline`]: the fill-vs-read rendezvous for one staging slot, and the [`Sync`] strategy
//! deduced from the operands' delivery. The [`Barrier`](Sync::Barrier) strategy mirrors
//! cubek-matmul's `specialized/matmul.rs`; [`Cube`](Sync::Cube) and [`Solo`](Sync::Solo) are
//! degenerate cases.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use crate::*;

/// How a slot rendezvouses its fill against its read; fixed comptime at construction
/// from the operands' delivery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Sync {
    /// One unit fills and reads its own slot: no collective (single-plane / CPU).
    Solo,
    /// Cooperative element copy rendezvoused on one cube-wide `sync_cube` per phase. The sync sits
    /// in `write` and covers both this slot's fill→read and the sibling's read→refill.
    Cube,
    /// Hardware async bulk copy (TMA): `full`/`empty` mbarrier pair with a `phase` parity, producer
    /// and consumer decoupled so the copy overlaps compute.
    Barrier,
}

impl Sync {
    /// Join the rendezvous requirements of a slot's sources, over a walk `fillers` planes fill.
    /// `Barrier` dominates `Cube` because TMA transaction completion must be included in the
    /// slot's publication, and a filled slot starts there: `Cube` rendezvouses on `sync_cube`,
    /// which needs every unit of the cube, and the two roles never meet there.
    pub(crate) fn for_deliveries(deliveries: &[Delivery], fillers: usize) -> Sync {
        assert!(
            !deliveries.is_empty(),
            "Staging: a slot must have at least one delivery"
        );
        let floor = if fillers > 0 { Sync::Barrier } else { Sync::Cube };
        deliveries.iter().fold(floor, |sync, delivery| {
            match (sync, delivery.rendezvous()) {
                (Sync::Barrier, _) | (_, Sync::Barrier) => Sync::Barrier,
                (Sync::Cube, Sync::Cube) => Sync::Cube,
                (Sync::Solo, _) | (_, Sync::Solo) => {
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
            .any(|delivery| delivery.rendezvous() == Sync::Cube)
    }
}

/// The rendezvous for one slot, and every barrier it owns. The acquire/release operations live
/// on [`Staging`]; [`fill`](Pipeline::fill) is the one op a `write` body reaches for directly.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub enum Pipeline {
    /// Synchronous cooperative element copy, rendezvoused on one `sync_cube` per phase.
    /// The variant (not a flag) carries the choice, so the dispatch is comptime and the
    /// rendezvous emits a bare barrier, never a branch-wrapped one.
    Cube,
    /// A single unit fills and reads its own slot: no collective at all.
    Solo,
    /// Async producer/consumer decoupled over a `full`/`empty` mbarrier pair, one parity each,
    /// so the fill overlaps compute. TMA motivates it, but the barrier itself is
    /// delivery-agnostic; see [`Pipeline::fill`].
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
        /// A mixed TMA/synchronous slot needs every producer to publish; a pure TMA slot is
        /// published by its elected issuer alone.
        collective_full: bool,
        /// The one unit that issues this slot's bulk copies and declares their bytes
        /// ([`Pipeline::elected`]).
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
impl Pipeline {
    /// Allocate the pipeline for `sync`: the `full`/`empty` mbarrier pair, sealed by a proxy fence
    /// before any bulk copy, for [`Barrier`](Sync::Barrier); nothing to allocate otherwise.
    ///
    /// Both barriers are armed and fenced before any plane takes a role, which is why the
    /// election here is unit 0 and the `sync_cube` is the whole cube's: every unit is still
    /// present.
    pub(crate) fn new(
        #[comptime] sync: Sync,
        #[comptime] collective_full: bool,
        #[comptime] fillers: usize,
    ) -> Pipeline {
        match sync {
            Sync::Solo => Pipeline::new_Solo(),
            Sync::Cube => Pipeline::new_Cube(),
            Sync::Barrier => {
                let full = Barrier::shared(Pipeline::producers(collective_full), UNIT_POS == 0);
                let empty = Barrier::shared(Pipeline::consumers(fillers), UNIT_POS == 0);
                sync_async_proxy_shared();
                sync_cube();
                let elected = Pipeline::elected(fillers);
                Pipeline::new_Barrier(full, empty, collective_full, elected, 0, 0)
            }
        }
    }

    /// Units that arrive on `full`. A pure TMA slot is published by the one unit that issued its
    /// copy; a mixed slot also holds a cooperative fill, so every unit that wrote it publishes.
    pub fn producers(#[comptime] collective_full: bool) -> u32 {
        if comptime!(collective_full) {
            CUBE_DIM
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
            Pipeline::consumers(fillers)
        }
    }

    /// Fill staged `dst` from `src`, the one operation a `fill` body performs. A `Barrier` slot
    /// stages under its `full` mbarrier; a `Cube` slot is a plain blocking
    /// [`copy_from`](Tile::copy_from). In-place operands never reach it: they allocate no
    /// destination, so their read goes to the source instead of through a fill.
    pub fn fill<E: Numeric>(&self, dst: &mut Tile<E>, src: &Tile<E>) {
        // Bound before the match, which borrows the kind: the fill needs the logical space both
        // sides carry (a gathered source is addressed per axis).
        let space = comptime!(dst.space.clone());
        match self {
            Pipeline::Barrier { full, elected, .. } => match (&mut dst.tile_kind, &src.tile_kind) {
                (TileKind::Smem(d), TileKind::TmaGmem(s)) => {
                    // One issuer, and the same unit that declares the bytes: the transaction
                    // count is that unit's alone, so a second issuer would over-count the stage.
                    if UNIT_POS == *elected {
                        full.expect_tx(d.size_bytes());
                        s.stage_into(d, full);
                    }
                }
                // A strided source under a barrier is a plain synchronous copy.
                (TileKind::Smem(d), TileKind::Gmem(s) | TileKind::Smem(s)) => d.fill_from(s, space),
                (TileKind::Smem(d), TileKind::Procedural(s)) => d.fill_procedural(s, space),
                _ => panic!("Pipeline::fill: unsupported kind pairing"),
            },
            Pipeline::Cube | Pipeline::Solo => dst.copy_from(src),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_and_strided_share_a_cube_pipeline() {
        assert_eq!(
            Sync::for_deliveries(&[Delivery::Procedural, Delivery::Copy], 0),
            Sync::Cube
        );
    }

    #[test]
    fn procedural_and_tma_share_a_barrier_pipeline() {
        assert_eq!(
            Sync::for_deliveries(&[Delivery::Procedural, Delivery::Tma], 0),
            Sync::Barrier
        );
        assert!(Sync::collective_full(&[
            Delivery::Procedural,
            Delivery::Tma
        ]));
    }

    #[test]
    fn pure_tma_keeps_its_single_producer_arrival() {
        assert!(!Sync::collective_full(&[Delivery::Tma]));
    }

    /// `sync_cube` needs every unit of the cube, and a walk that sets planes aside to fill has
    /// none of its slots reached by all of them.
    #[test]
    fn a_filled_slot_rendezvouses_on_a_barrier_whatever_delivered_it() {
        assert_eq!(Sync::for_deliveries(&[Delivery::Copy], 1), Sync::Barrier);
    }
}

