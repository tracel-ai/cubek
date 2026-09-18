//! The [`Pipeline`]: the fill-vs-read rendezvous for one staging slot, and the [`Sync`] strategy
//! deduced from when the operands' fills complete ([`Completion`]). The
//! [`Barrier`](Sync::Barrier) strategy mirrors cubek-matmul's `specialized/matmul.rs`;
//! [`Cube`](Sync::Cube) and [`Solo`](Sync::Solo) are degenerate cases.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use crate::*;

/// When a fill of one delivery is complete, and so what publishes it: the one fact a slot asks
/// of a delivery, from which its rendezvous ([`Sync`]) and its publishers ([`Publishers`]) follow.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Completion {
    /// The unit's own stores: ordered by the rendezvous (`sync_cube`, or the unit's `arrive`).
    Stores,
    /// Copies the unit issued that land on their own: it commits them to the barrier, then
    /// arrives.
    IssuedCopies,
    /// One bulk transaction whose bytes one issuer declares: that issuer's `arrive` alone.
    Transaction,
}

/// How a slot rendezvouses its fill against its read; fixed comptime at construction
/// from the operands' delivery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Sync {
    /// One unit fills and reads its own slot: no collective (single-plane / CPU).
    Solo,
    /// Cooperative element copy rendezvoused on one cube-wide `sync_cube` per phase. The sync sits
    /// in `write` and covers both this slot's fill→read and the sibling's read→refill.
    Cube,
    /// Asynchronous fills (a bulk copy, or copies the units issue): `full`/`empty` mbarrier pair
    /// with a `phase` parity, producer and consumer decoupled so the fill overlaps compute.
    ///
    /// Who meets over the pair is the barrier's own business, and nothing the other two arms
    /// have an answer for: the fields ride the variant rather than travelling beside it.
    Barrier {
        /// Which units arrive on `full` to publish the fill ([`Publishers`]).
        publishers: Publishers,
        /// Planes of the cube set aside to fill this slot and take no tile
        /// ([`Level::filled_by`]). They are the reason a slot can need the barrier even when
        /// every fill of it completes by stores.
        fillers: usize,
        /// Whether any of the slot's fills is landed by the bulk-copy engine. That engine writes
        /// through the *async proxy*, so the barriers it signals have to be fenced into that
        /// proxy before it lands anything on them ([`Pipeline::new`]).
        transacts: bool,
    },
}

impl Sync {
    /// Join the rendezvous requirements of a slot's sources, over a walk `fillers` planes fill.
    /// A `sync_cube` orders stores and nothing else, so a fill that completes any other way
    /// needs the barrier; and a filled slot starts there: `Cube` rendezvouses on `sync_cube`,
    /// which needs every unit of the cube, and the two roles never meet there.
    pub(crate) fn for_deliveries(deliveries: &[Delivery], fillers: usize) -> Sync {
        assert!(
            !deliveries.is_empty(),
            "Staging: a slot must have at least one delivery"
        );
        let asynchronous = deliveries
            .iter()
            .any(|delivery| delivery.completion() != Completion::Stores);
        match fillers > 0 || asynchronous {
            true => Sync::Barrier {
                publishers: Publishers::of(deliveries, fillers),
                fillers,
                transacts: deliveries
                    .iter()
                    .any(|delivery| delivery.completion() == Completion::Transaction),
            },
            false => Sync::Cube,
        }
    }

    /// Whether this is the asynchronous rendezvous, the one an operand cannot be lifted out of.
    pub(crate) fn is_barrier(&self) -> bool {
        matches!(self, Sync::Barrier { .. })
    }

    /// Planes set aside to fill, which only a barrier slot has: a `sync_cube` needs every unit
    /// of the cube, so a walk that holds planes back never rendezvouses on one.
    pub(crate) fn fillers(&self) -> usize {
        match self {
            Sync::Barrier { fillers, .. } => *fillers,
            Sync::Cube | Sync::Solo => 0,
        }
    }
}

/// Which units arrive on a barrier slot's `full` to publish its fill ([`Pipeline::producers`]).
///
/// Every unit that wrote or issued into the slot has to be counted, and a unit that only issued
/// a bulk transaction has nothing of its own to publish: so a slot no plane was set aside for is
/// published by the one unit that issued its bulk copy, and by every unit as soon as any operand
/// completes by the units' own stores or issued copies. A slot filled by planes of their own
/// needs every one of their units whatever delivered it: nothing else holds a filling plane in
/// step with the elected one, and a plane that falls a lap behind waits on a parity that has
/// already gone by.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub(crate) enum Publishers {
    /// Every unit of the cube.
    Every,
    /// Every unit of the planes set aside to fill.
    Fillers,
    /// The one elected issuer alone.
    Elected,
}

impl Publishers {
    pub(crate) fn of(deliveries: &[Delivery], fillers: usize) -> Publishers {
        let units_own = deliveries
            .iter()
            .any(|delivery| delivery.completion() != Completion::Transaction);
        if fillers > 0 {
            Publishers::Fillers
        } else if units_own {
            Publishers::Every
        } else {
            Publishers::Elected
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
        /// Whether `full` counts every producer's arrival or only the elected issuer's
        /// ([`Publishers`]).
        all_publish: bool,
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
    /// Allocate the pipeline for `sync`: the `full`/`empty` mbarrier pair for
    /// [`Barrier`](Sync::Barrier); nothing to allocate otherwise.
    ///
    /// A slot the bulk-copy engine lands ([`transacts`](Sync::Barrier::transacts)) is fenced into
    /// the async proxy the engine writes through, before it is asked to land anything on these
    /// barriers. A copy the units issue writes through the generic proxy and needs no such fence
    /// — and the fence is Hopper's own instruction, so emitting it for a slot no engine fills is
    /// a kernel every earlier architecture refuses to assemble.
    ///
    /// Both barriers are armed before any plane takes a role, which is why the election here is
    /// unit 0 and the `sync_cube` is the whole cube's: every unit is still present.
    pub(crate) fn new(#[comptime] sync: Sync) -> Pipeline {
        match sync {
            Sync::Solo => Pipeline::new_Solo(),
            Sync::Cube => Pipeline::new_Cube(),
            Sync::Barrier {
                publishers,
                fillers,
                transacts,
            } => {
                let full = Barrier::shared(Pipeline::producers(publishers, fillers), UNIT_POS == 0);
                let empty = Barrier::shared(Pipeline::consumers(fillers), UNIT_POS == 0);
                if comptime!(transacts) {
                    sync_async_proxy_shared();
                }
                sync_cube();
                let elected = Pipeline::elected(fillers);
                let all_publish = comptime!(publishers != Publishers::Elected);
                Pipeline::new_Barrier(full, empty, all_publish, elected, 0, 0)
            }
        }
    }

    /// Units that arrive on `full`: the count [`Publishers`] names, over a walk `fillers` planes
    /// fill.
    ///
    /// The elected unit is one of the arrivals, and it declares the transaction bytes before it
    /// arrives, so the phase cannot complete on the others' arrivals with the bytes still to
    /// come.
    pub(crate) fn producers(#[comptime] publishers: Publishers, #[comptime] fillers: usize) -> u32 {
        match publishers {
            Publishers::Every => CUBE_DIM,
            Publishers::Fillers => comptime!(fillers as u32) * CUBE_DIM_X,
            Publishers::Elected => 1u32.runtime(),
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
    /// stages under its `full` mbarrier — the engine's bulk copy, or the units' own issued
    /// copies, as the source's delivery says; a `Cube` slot is a plain blocking
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
                (TileKind::Smem(d), TileKind::Gmem(s)) => match comptime!(s.access.delivery) {
                    // The unit issues the lines and reads none of them back; `release_write`
                    // commits what it issued to `full` before arriving on it.
                    Delivery::AsyncCopy => d.fill_async(s, space),
                    // A stored source under a barrier is a plain synchronous copy.
                    Delivery::Copy => d.fill_from(s, space),
                    other => {
                        panic!("Pipeline::fill: a bound tensor is never delivered by {other:?}")
                    }
                },
                // Shared memory is on chip already: nothing asynchronous moves it.
                (TileKind::Smem(d), TileKind::Smem(s)) => d.fill_from(s, space),
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
        assert!(Sync::for_deliveries(&[Delivery::Procedural, Delivery::Tma], 0).is_barrier());
        assert_eq!(
            Publishers::of(&[Delivery::Procedural, Delivery::Tma], 0),
            Publishers::Every
        );
    }

    #[test]
    fn pure_tma_keeps_its_single_producer_arrival() {
        assert_eq!(Publishers::of(&[Delivery::Tma], 0), Publishers::Elected);
    }

    /// A slot filled by planes of their own is published by every one of their units, whatever
    /// delivered it: the count is what holds a lagging filling plane in step.
    #[test]
    fn a_filled_slot_is_published_by_every_filling_unit() {
        assert_eq!(Publishers::of(&[Delivery::Tma], 2), Publishers::Fillers);
    }

    /// `sync_cube` needs every unit of the cube, and a walk that sets planes aside to fill has
    /// none of its slots reached by all of them.
    #[test]
    fn a_filled_slot_rendezvouses_on_a_barrier_whatever_delivered_it() {
        assert!(Sync::for_deliveries(&[Delivery::Copy], 1).is_barrier());
    }

    /// The proxy fence follows the bulk copy rather than the barrier.
    ///
    /// The engine writes through the async proxy and the cube's own units do not, so only a slot
    /// the engine fills has to fence its barriers into it. The fence is also the newest
    /// architectures' own instruction, so a slot that carried one it did not need would be a
    /// kernel every earlier one refuses to assemble — and a kernel that never assembles reads
    /// back as whatever its output already held.
    #[test]
    fn only_a_bulk_transaction_fences_the_slots_barriers() {
        let fences = |delivery| {
            matches!(
                Sync::for_deliveries(&[delivery, delivery], 0),
                Sync::Barrier {
                    transacts: true,
                    ..
                }
            )
        };
        assert!(fences(Delivery::Tma));
        assert!(!fences(Delivery::AsyncCopy));
        assert!(!fences(Delivery::Copy));
    }
}
