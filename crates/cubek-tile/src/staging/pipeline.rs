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
    /// Resolve one slot's synchronization from all of its source deliveries. Strided copies and
    /// procedural materialization are cooperative, so they share [`Cube`](Self::Cube). A slot
    /// containing only TMA sources uses [`Barrier`](Self::Barrier). TMA cannot yet share a slot
    /// with a synchronous source: publishing its barrier would need another cube rendezvous.
    pub fn for_deliveries(deliveries: &[Delivery]) -> Sync {
        assert!(
            !deliveries.is_empty(),
            "Staging: a slot must have at least one delivery"
        );
        let has_tma = deliveries.contains(&Delivery::Tma);
        if has_tma {
            assert!(
                deliveries.iter().all(|delivery| *delivery == Delivery::Tma),
                "Staging: a TMA source cannot share a slot with a synchronous strided or procedural source"
            );
            Sync::Barrier
        } else {
            Sync::Cube
        }
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
        /// Producer→consumer (one producer arrival): flips once the fill's transaction bytes land.
        full: Shared<Barrier>,
        /// Consumer→producer (one arrival per unit): flips once every unit has read and freed the slot.
        empty: Shared<Barrier>,
        /// mbarrier parity for `wait_parity`; flipped once per read.
        phase: u32,
    },
}

#[cube]
impl Pipeline {
    /// Allocate the pipeline for `sync`: the `full`/`empty` mbarrier pair, sealed by a proxy fence
    /// before any bulk copy, for [`Barrier`](Sync::Barrier); nothing to allocate otherwise.
    pub(crate) fn new(#[comptime] sync: Sync) -> Pipeline {
        match sync {
            Sync::Solo => Pipeline::new_Solo(),
            Sync::Cube => Pipeline::new_Cube(),
            Sync::Barrier => {
                // full: one producer arrival; empty: one arrival per unit.
                let full = Barrier::shared(1, UNIT_POS == 0);
                let empty = Barrier::shared(CUBE_DIM, UNIT_POS == 0);
                sync_async_proxy_shared();
                sync_cube();
                Pipeline::new_Barrier(full, empty, 0)
            }
        }
    }

    /// Fill staged `dst` from `src`, the one operation a `fill` body performs. A `Barrier` slot
    /// stages under its `full` mbarrier; a `Cube` slot is a plain blocking
    /// [`copy_from`](Tile::copy_from).
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
            Sync::for_deliveries(&[Delivery::Procedural, Delivery::Strided]),
            Sync::Cube
        );
    }

    #[test]
    #[should_panic(expected = "TMA source cannot share a slot")]
    fn procedural_and_tma_are_explicitly_rejected() {
        Sync::for_deliveries(&[Delivery::Procedural, Delivery::Tma]);
    }
}
