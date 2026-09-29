//! The [`Meeting`]: the fill-vs-read rendezvous for one staging slot, and the [`Rendezvous`] strategy
//! deduced from the operands' delivery. [`Barrier`](Rendezvous::Barrier) mirrors cubek-matmul's
//! `specialized/matmul.rs`; [`Cube`](Rendezvous::Cube) is the degenerate case.

use cubecl::prelude::barrier::Barrier;
use cubecl::prelude::*;

use crate::*;

/// How a slot rendezvouses its fill against its read; fixed comptime at construction
/// from the operands' delivery.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Rendezvous {
    /// Cooperative element copy rendezvoused on one cube-wide `sync_cube` per phase. The sync sits
    /// in `write` and covers both this slot's fill→read and the sibling's read→refill.
    Cube,
    /// [`Cube`](Rendezvous::Cube) for a stage one plane owns: its own units fill it and read it,
    /// so one `sync_plane` per phase meets them all and no other plane waits.
    ///
    /// Every plane of the cube still has to reach the same count of them: a backend with no
    /// plane-wide barrier (WGSL) lowers `sync_plane` to the workgroup's, which every unit of the
    /// cube must meet.
    Plane,
    /// Hardware async bulk copy (TMA): `full`/`empty` mbarrier pair with a `phase` parity, producer
    /// and consumer decoupled so the copy overlaps compute.
    Barrier,
}

impl Rendezvous {
    /// Join the rendezvous requirements of a slot's sources, over a walk `fillers` planes fill,
    /// for stages `owner` holds. `Barrier` dominates `Cube` because TMA transaction completion
    /// must be in the slot's publication, while `Cube` rendezvouses on `sync_cube`, where the two
    /// roles never meet. A plane's own stages meet on `sync_plane`.
    ///
    /// # Panics
    ///
    /// A plane's own stages filled by a bulk copy, or by planes set aside to fill: both meet on
    /// barriers the whole cube arms, which a plane walking its own stages never shares.
    pub(crate) fn for_deliveries(
        deliveries: &[Delivery],
        fillers: usize,
        owner: StageOwner,
    ) -> Rendezvous {
        assert!(
            !deliveries.is_empty(),
            "Slot: a slot must have at least one delivery"
        );
        if let StageOwner::Plane { .. } = owner {
            assert!(
                fillers == 0
                    && deliveries
                        .iter()
                        .all(|delivery| delivery.rendezvous() == Rendezvous::Cube),
                "Slot: a stage one plane owns is filled by that plane's units alone, so neither a \
                 bulk copy nor planes set aside to fill can deliver it"
            );
            return Rendezvous::Plane;
        }
        let floor = if fillers > 0 {
            Rendezvous::Barrier
        } else {
            Rendezvous::Cube
        };
        deliveries.iter().fold(floor, |sync, delivery| {
            match (sync, delivery.rendezvous()) {
                (Rendezvous::Barrier, _) | (_, Rendezvous::Barrier) => Rendezvous::Barrier,
                (Rendezvous::Cube, Rendezvous::Cube) => Rendezvous::Cube,
                (Rendezvous::Plane, _) | (_, Rendezvous::Plane) => {
                    unreachable!("a delivery asks for the cube or a barrier, never a plane")
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
    /// [`Cube`](Meeting::Cube) for a stage one plane owns, rendezvoused on `sync_plane`
    /// ([`Rendezvous::Plane`]).
    Plane,
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
            Rendezvous::Cube => Meeting::new_Cube(),
            Rendezvous::Plane => Meeting::new_Plane(),
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
    /// stages under `full`, a `Cube` or a `Plane` slot is a blocking [`copy_from`](Tile::copy_from),
    /// spread over the units `dst` names ([`Access::workers`](crate::Access));
    /// in-place operands allocate no destination and never reach it, reading the source instead.
    pub fn fill<E: Numeric>(&self, dst: &mut Tile<E>, src: &Tile<E>) {
        // Bound before the match, which borrows the kind: the fill needs the logical space both
        // sides carry (a gathered source is addressed per axis).
        let space = comptime!(dst.place.space.clone());
        // A source carrying scales or a table decodes where it is copied, whichever meeting.
        let decodes = src.scaled();
        if comptime!(decodes) {
            dst.copy_from(src);
        } else {
            self.fill_as_it_lies(dst, src, space);
        }
    }

    /// [`fill`](Meeting::fill) a source that decodes nothing: its values as they lie.
    fn fill_as_it_lies<E: Numeric>(
        &self,
        dst: &mut Tile<E>,
        src: &Tile<E>,
        #[comptime] space: Space,
    ) {
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
            Meeting::Cube | Meeting::Plane => dst.copy_from(src),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn procedural_and_strided_share_a_cube_pipeline() {
        assert_eq!(
            Rendezvous::for_deliveries(
                &[Delivery::Procedural, Delivery::Copy],
                0,
                StageOwner::Cube
            ),
            Rendezvous::Cube
        );
    }

    #[test]
    fn procedural_and_tma_share_a_barrier_pipeline() {
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Procedural, Delivery::Tma], 0, StageOwner::Cube),
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
            Rendezvous::for_deliveries(&[Delivery::Copy], 1, StageOwner::Cube),
            Rendezvous::Barrier
        );
    }

    /// A plane's own stages meet on its own barrier, whatever copy fills them.
    #[test]
    fn a_planes_own_stage_rendezvouses_on_the_plane() {
        let owner = StageOwner::Plane { planes: 4 };
        assert_eq!(
            Rendezvous::for_deliveries(&[Delivery::Procedural, Delivery::Copy], 0, owner),
            Rendezvous::Plane
        );
    }

    /// A bulk copy's barrier is armed by the whole cube, which a plane walking its own stages
    /// never meets.
    #[test]
    #[should_panic(expected = "filled by that plane's units alone")]
    fn a_planes_own_stage_takes_no_bulk_copy() {
        Rendezvous::for_deliveries(&[Delivery::Tma], 0, StageOwner::Plane { planes: 4 });
    }

    /// Planes set aside to fill are planes the owning plane's units are not.
    #[test]
    #[should_panic(expected = "filled by that plane's units alone")]
    fn a_planes_own_stage_takes_no_filling_planes() {
        Rendezvous::for_deliveries(&[Delivery::Copy], 1, StageOwner::Plane { planes: 4 });
    }
}
