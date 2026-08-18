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
    /// Join the rendezvous requirements of a slot's sources. `Barrier` dominates `Cube` because
    /// TMA transaction completion must be included in the slot's publication.
    pub fn for_deliveries(deliveries: &[Delivery]) -> Sync {
        assert!(
            !deliveries.is_empty(),
            "Staging: a slot must have at least one delivery"
        );
        deliveries.iter().fold(Sync::Cube, |sync, delivery| {
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
    pub fn collective_full(deliveries: &[Delivery]) -> bool {
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
    /// Async producer/consumer decoupled over a `full`/`empty` mbarrier pair with a `phase`
    /// parity, so the fill overlaps compute. TMA motivates it, but the barrier itself is
    /// delivery-agnostic; see [`Pipeline::fill`].
    Barrier {
        /// Producer→consumer: flips after every unit arrives and all declared TMA transaction
        /// bytes land.
        full: Shared<Barrier>,
        /// Consumer→producer (one arrival per unit): flips once every unit has read and freed the slot.
        empty: Shared<Barrier>,
        /// A mixed TMA/synchronous slot needs every producer to publish; a pure TMA slot is
        /// published by its elected issuer alone.
        #[cube(comptime)]
        collective_full: bool,
        /// mbarrier parity for `wait_parity`; flipped once per read.
        phase: u32,
    },
}

#[cube]
impl Pipeline {
    /// Allocate the pipeline for `sync`: the `full`/`empty` mbarrier pair, sealed by a proxy fence
    /// before any bulk copy, for [`Barrier`](Sync::Barrier); nothing to allocate otherwise.
    pub(crate) fn new(#[comptime] sync: Sync, #[comptime] collective_full: bool) -> Pipeline {
        match sync {
            Sync::Solo => Pipeline::new_Solo(),
            Sync::Cube => Pipeline::new_Cube(),
            Sync::Barrier => {
                // A mixed slot collects cooperative writers and TMA bytes; pure TMA keeps its
                // one elected producer arrival.
                let full_arrivals = comptime!(if collective_full { CUBE_DIM } else { 1u32 });
                let full = Barrier::shared(full_arrivals, UNIT_POS == 0);
                let empty = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
                sync_async_proxy_shared();
                sync_cube();
                Pipeline::new_Barrier(full, empty, collective_full, 0)
            }
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
            Pipeline::Barrier { full, .. } => match (&mut dst.tile_kind, &src.tile_kind) {
                (TileKind::Smem(d), TileKind::TmaGmem(s)) => {
                    if UNIT_POS == 0 {
                        full.expect_tx(d.size_bytes());
                    }
                    s.stage_into(d, full);
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
            Sync::for_deliveries(&[Delivery::Procedural, Delivery::Copy]),
            Sync::Cube
        );
    }

    #[test]
    fn procedural_and_tma_share_a_barrier_pipeline() {
        assert_eq!(
            Sync::for_deliveries(&[Delivery::Procedural, Delivery::Tma]),
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
}
