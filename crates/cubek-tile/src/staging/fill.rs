//! Filling a staging slot with matmul operands: the operand-deducing [`new`](Staging::new), the
//! pin split ([`fill_pinned`](Staging::fill_pinned) / [`fill_streamed`](Staging::fill_streamed)),
//! and the closure-driven [`fill`](Staging::fill) / [`consume`](Staging::consume) with their
//! hand-written expands.
//!
//! `fill`/`consume` are hand-written expand methods because a `Drop` guard can't emit a barrier
//! op in cubecl and `#[cube]` rejects `impl Trait` args.

use core::option::Option;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// The slot-wide sync and whether all units publish its full barrier, derived only from operands
/// that physically materialize.
fn slot_plan(deliveries: &[Delivery]) -> (Sync, bool) {
    let sync = if deliveries.is_empty() {
        Sync::Solo
    } else {
        Sync::for_deliveries(deliveries)
    };
    (sync, Sync::collective_full(deliveries))
}

fn rendezvous_deliveries(residences: &[Residence], deliveries: &[Delivery]) -> Vec<Delivery> {
    residences
        .iter()
        .zip(deliveries)
        .filter_map(|(residence, delivery)| (*residence == Residence::Smem).then_some(*delivery))
        .collect()
}

/// A plane partition is private to its unit and assumes a solo fill. It cannot share a slot with
/// a shared-memory stage, whose transfer selects a slot-wide Cube or Barrier pipeline.
fn compatible_slot_residences(residences: &[Residence]) -> bool {
    let has_plane = residences.contains(&Residence::Plane);
    let has_smem = residences.contains(&Residence::Smem);
    !(has_plane && has_smem)
}

/// A slot builds a payload for every operand it holds, and an operand a level above already
/// materialized into plane-private registers has none to build: it can be neither filled through
/// the pipeline nor rebound per region. Such a level has to leave every operand
/// [`InPlace`](Residence::InPlace), which lowers to the plain recursive walk and builds no slot at
/// all, so reaching here means one operand asked to materialize beside a fragment.
fn slot_admits_operands(fragments: &[bool]) -> bool {
    !fragments.contains(&true)
}

/// Fill a materialized operand through its slot pipeline, or rebind an in-place payload to the
/// source region. The pipeline is slot-wide while this decision is per-operand, which is what lets
/// one slot hold a staged operand beside one read where it lies.
#[cube]
pub(crate) fn fill_operand<T: Numeric>(
    dst: &mut Tile<T>,
    src: &Tile<T>,
    #[comptime] residence: Residence,
    pipe: &Pipeline,
) {
    if comptime!(residence == Residence::InPlace) {
        dst.rebind_from(src);
    } else {
        pipe.fill(dst, src);
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Build a slot staging one region of the operands `lhs`/`rhs`. An [`Residence::Plane`]
    /// stages into plane-private tile partitions ([`Solo`](Sync::Solo)); [`Smem`](Residence::Smem)
    /// takes fresh shared memory, with [`Sync`] deduced from the operands' delivery.
    pub fn new(
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
    ) -> Staging<(Tile<Lhs>, Tile<Rhs>)> {
        // Admission first: an operand's delivery is only meaningful once it is known to be a fill
        // source at all, and a resident fragment never is.
        let lhs_fragment = lhs.resident_fragment();
        let rhs_fragment = rhs.resident_fragment();
        comptime!(assert!(
            slot_admits_operands(&[lhs_fragment, rhs_fragment]),
            "Staging: an operand already materialized as a plane fragment cannot share a slot; \
             the other operand asked to materialize at this level, so give it Residence::InPlace \
             here (materializing it at a level above instead)"
        ));
        let lhs_residence = lhs.residence(comptime!(&out));
        let rhs_residence = rhs.residence(comptime!(&out));
        comptime!(assert!(
            compatible_slot_residences(&[lhs_residence, rhs_residence]),
            "Staging: Plane and Smem operands cannot share a slot"
        ));
        let lhs_delivery = lhs.delivery();
        let rhs_delivery = rhs.delivery();
        // Pin an operand only when its window is genuinely fixed across the walk. A barrier
        // pipeline arrives `full` once per fill, so a TMA pair keeps the joint per-region fill;
        // splitting an invariant out would corrupt its phase. A dynamic level can't decide
        // invariance at comptime. Both fall back to streaming (pin = false).
        let split =
            comptime!(op_space.is_static() && !lhs_delivery.is_tma() && !rhs_delivery.is_tma());
        let pin_lhs = comptime!(split && op_space.walk_invariant(&lhs.space));
        let pin_rhs = comptime!(split && op_space.walk_invariant(&rhs.space));
        let materialized = comptime!(rendezvous_deliveries(
            &[lhs_residence, rhs_residence],
            &[lhs_delivery, rhs_delivery],
        ));
        let (sync, collective_full) = comptime!(slot_plan(&materialized));
        let stages = (
            stage_operand(lhs, comptime!(out.clone()), lhs_residence),
            stage_operand(rhs, comptime!(out.clone()), rhs_residence),
        );
        Staging::wrap(
            stages,
            Pipeline::new(sync, collective_full),
            pin_lhs,
            pin_rhs,
            lhs_residence,
            comptime!(Option::Some(rhs_residence)),
        )
    }

    /// Fill the pinned operand(s), those the walk leaves invariant, from `region`'s window.
    /// Their window never moves, so `region` is region 0 and this runs once, above the loop.
    /// A no-op when nothing is pinned (both operands stream).
    pub fn fill_pinned(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let pin_lhs = comptime!(self.pin_lhs);
        let pin_rhs = comptime!(self.pin_rhs);
        let lhs_residence = self.residence();
        let rhs_residence = self.residence_rhs();
        if comptime!(pin_lhs || pin_rhs) {
            self.fill(|staged_operands, pipe| {
                if comptime!(pin_lhs) {
                    let source = lhs.at(region);
                    fill_operand(&mut staged_operands.0, &source, lhs_residence, pipe);
                }
                if comptime!(pin_rhs) {
                    let source = rhs.at(region);
                    fill_operand(&mut staged_operands.1, &source, rhs_residence, pipe);
                }
            });
        }
    }

    /// Fill the streamed operand(s), everything not pinned, from `region`'s window. Runs per
    /// region inside the walk.
    pub fn fill_streamed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let pin_lhs = comptime!(self.pin_lhs);
        let pin_rhs = comptime!(self.pin_rhs);
        let lhs_residence = self.residence();
        let rhs_residence = self.residence_rhs();
        self.fill(|staged_operands, pipe| {
            if comptime!(!pin_lhs) {
                let source = lhs.at(region);
                fill_operand(&mut staged_operands.0, &source, lhs_residence, pipe);
            }
            if comptime!(!pin_rhs) {
                let source = rhs.at(region);
                fill_operand(&mut staged_operands.1, &source, rhs_residence, pipe);
            }
        });
    }
}

// `fill`/`consume` take closures so the body stays caller-defined (fill each buffer however, run the
// mma). They're provided for the `(Tile<Lhs>, Tile<Rhs>)` payload (not generic `T`): closure-parameter
// inference can't resolve the projection `&mut T::ExpandType` through a generic `T`, but resolves the
// spelled-out tiles fine.
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Producer: wait the slot is free, run `fill` over the staged buffers and the slot's
    /// [`Pipeline`], then publish. See [`StagingExpand::__expand_fill_method`].
    pub fn fill(&mut self, _fill: impl FnOnce(&mut (Tile<Lhs>, Tile<Rhs>), &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the two staged tiles to `compute`, then free the slot.
    /// Each tile's bytes and runtime map were stored together by the producer.
    /// See [`StagingExpand::__expand_consume_method`].
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }

    /// Consumer for a fill no later fill will publish (the walk's final regions): publish
    /// the slot first, then consume. See [`StagingExpand::__expand_consume_final_method`].
    pub fn consume_final(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagingExpand<(Tile<Lhs>, Tile<Rhs>)> {
    pub fn __expand_fill_method<F>(&mut self, scope: &Scope, fill: F)
    where
        F: FnOnce(&Scope, &mut (TileExpand<Lhs>, TileExpand<Rhs>), &PipelineExpand),
    {
        self.__expand_acquire_write_method(scope);
        fill(scope, &mut self.data, &self.pipeline);
        self.__expand_release_write_method(scope);
    }

    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<Lhs>, &TileExpand<Rhs>),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data.0, &self.data.1);
        self.__expand_release_read_method(scope);
    }

    pub fn __expand_consume_final_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<Lhs>, &TileExpand<Rhs>),
    {
        self.__expand_publish_method(scope);
        self.__expand_consume_method(scope, compute);
    }
}

#[cube]
impl<T: Numeric> Staging<Tile<T>> {
    /// Build a slot staging one region of operand `input`.
    pub fn single(
        input: &Tile<T>,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
    ) -> Staging<Tile<T>> {
        let fragment = input.resident_fragment();
        comptime!(assert!(
            slot_admits_operands(&[fragment]),
            "Staging: an operand already materialized as a plane fragment cannot be held by a \
             slot; give it Residence::InPlace at this level"
        ));
        let residence = input.residence(comptime!(&out));
        let delivery = input.delivery();
        // Pin only when the window is genuinely fixed across the walk: a TMA pair keeps the joint
        // per-region fill (its barrier pipeline arrives `full` once per fill), and a dynamic level
        // can't decide invariance at comptime. Both fall back to streaming (pin = false).
        let split = comptime!(op_space.is_static() && !delivery.is_tma());
        let pin = comptime!(split && op_space.walk_invariant(&input.space));
        let materialized = comptime!(rendezvous_deliveries(&[residence], &[delivery]));
        let (sync, collective_full) = comptime!(slot_plan(&materialized));
        let staged_input = stage_operand(input, comptime!(out.clone()), residence);
        Staging::wrap(
            staged_input,
            Pipeline::new(sync, collective_full),
            pin,
            false,
            residence,
            comptime!(Option::None),
        )
    }

    /// Fill the pinned operand from `region`'s window.
    pub fn fill_pinned(&mut self, input: &Tile<T>, region: &Region) {
        let pin = self.pinned();
        let residence = self.residence();
        if comptime!(pin) {
            self.fill(|s, pipe| {
                let source = input.at(region);
                fill_operand(s, &source, residence, pipe);
            });
        }
    }

    /// Fill the streamed operand from `region`'s window.
    pub fn fill_streamed(&mut self, input: &Tile<T>, region: &Region) {
        let pin = self.pinned();
        let residence = self.residence();
        self.fill(|s, pipe| {
            if comptime!(!pin) {
                let source = input.at(region);
                fill_operand(s, &source, residence, pipe);
            }
        });
    }

    /// Whether the sole operand is pinned. `pin_lhs` is the `(Tile, Tile)` payload's field name;
    /// a unary payload has no `rhs` to pair it with, so this names what the field means here
    /// instead of reading a `lhs` that isn't one.
    fn pinned(&self) -> comptime_type!(bool) {
        comptime!(self.pin_lhs)
    }
}

impl<T: Numeric> Staging<Tile<T>> {
    /// Producer: wait the slot is free, run `fill` over the staged buffer and the slot's
    /// [`Pipeline`], then publish.
    pub fn fill(&mut self, _fill: impl FnOnce(&mut Tile<T>, &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the staged tile to `compute`, then free the slot.
    /// Each tile's bytes and runtime map were stored together by the producer.
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<T>)) {
        unexpanded!()
    }

    /// Consumer for a fill no later fill will publish: publish the slot first, then consume.
    pub fn consume_final(&mut self, _compute: impl FnOnce(&Tile<T>)) {
        unexpanded!()
    }
}

impl<T: Numeric> StagingExpand<Tile<T>> {
    pub fn __expand_fill_method<F>(&mut self, scope: &Scope, fill: F)
    where
        F: FnOnce(&Scope, &mut TileExpand<T>, &PipelineExpand),
    {
        self.__expand_acquire_write_method(scope);
        fill(scope, &mut self.data, &self.pipeline);
        self.__expand_release_write_method(scope);
    }

    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<T>),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data);
        self.__expand_release_read_method(scope);
    }

    pub fn __expand_consume_final_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<T>),
    {
        self.__expand_publish_method(scope);
        self.__expand_consume_method(scope, compute);
    }
}

/// Allocate one staged operand for `residence`. A gathered operand keeps its compacted physical
/// window and projection, so staging does not replicate each logical element for every gather
/// tap. The leaf performs the gather on read instead; that keeps staging compact but means
/// adjacent logical regions may re-read overlapping halo cells.
#[cube]
fn stage_operand<T: Numeric>(
    input: &Tile<T>,
    #[comptime] out: Space,
    #[comptime] residence: Residence,
) -> Tile<T> {
    let gathered = input.gathered();
    let delivery = input.delivery();
    match comptime!(residence) {
        // Nothing is allocated: the payload names the same bytes (or the same recipe) as the
        // source, one level down, and each region rebinds it rather than filling it.
        Residence::InPlace => match &input.tile_kind {
            TileKind::Procedural(data) => Tile::<T> {
                tile_kind: TileKind::new_Procedural(data.at_space(comptime!(input.space.divide()))),
                space: comptime!(input.space.divide()),
                leaf: comptime!(input.leaf),
            },
            TileKind::Gmem(data) => Tile::<T> {
                tile_kind: TileKind::new_Gmem(data.in_place_slot(comptime!(input.space.clone()))),
                space: comptime!(input.space.divide()),
                leaf: comptime!(input.leaf),
            },
            TileKind::Smem(data) => Tile::<T> {
                tile_kind: TileKind::new_Smem(data.in_place_slot(comptime!(input.space.clone()))),
                space: comptime!(input.space.divide()),
                leaf: comptime!(input.leaf),
            },
            // A tensor map is not element-addressable: there is no window to hand down, and its
            // only sink is a hardware bulk copy into shared memory.
            TileKind::TmaGmem(_) => {
                panic!("Staging: a TMA source cannot be read in place; give it Residence::Smem")
            }
            // Unreachable: `slot_admits_operands` turns a resident fragment away before any
            // payload is built, since it has no window to rebind.
            TileKind::PlaneTile(_) | TileKind::PlanePartition(_) => {
                panic!("Staging: a resident fragment cannot be held by a slot")
            }
        },
        Residence::Plane => {
            comptime!(assert!(
                !delivery.is_tma(),
                "Staging: a TMA source cannot stage into plane tiles"
            ));
            comptime!(assert!(
                !gathered,
                "Staging: a gathered operand cannot stage into plane tiles (Residence::Plane); \
                 only Residence::Smem stages one, as the compacted window its leaf reads"
            ));
            PlanePartition::store(
                comptime!(input.space.divide()),
                comptime!(input.leaf),
                comptime!(out.clone()),
            )
        }
        Residence::Smem => MemData::smem_like(input),
        // `Tile::residence` resolves it against the level below before anything reaches here.
        Residence::Auto => panic!("Staging: Residence::Auto is a request, not a backing"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_in_place_operand_does_not_join_the_slot_rendezvous() {
        let deliveries = rendezvous_deliveries(
            &[Residence::InPlace, Residence::Smem],
            &[Delivery::Procedural, Delivery::Copy],
        );
        assert_eq!(deliveries, vec![Delivery::Copy]);
        assert_eq!(slot_plan(&deliveries).0, Sync::Cube);
    }

    #[test]
    fn a_plane_stage_cannot_share_a_slot_with_smem() {
        assert!(!compatible_slot_residences(&[
            Residence::Plane,
            Residence::Smem,
        ]));
        assert!(!compatible_slot_residences(&[
            Residence::Smem,
            Residence::Plane,
        ]));
        assert!(compatible_slot_residences(&[
            Residence::Plane,
            Residence::InPlace,
        ]));
    }

    /// An asymmetric plan can leave one operand a fragment while the other still asks to
    /// materialize. Both read `InPlace` at that point, so the residences alone look compatible;
    /// only the physical kind tells the pair apart.
    #[test]
    fn a_resident_fragment_is_turned_away_from_a_slot() {
        assert!(!slot_admits_operands(&[true, false]));
        assert!(!slot_admits_operands(&[false, true]));
        assert!(!slot_admits_operands(&[true]));
        assert!(slot_admits_operands(&[false, false]));
        // The residence check on its own waves the same pair through.
        assert!(compatible_slot_residences(&[
            Residence::InPlace,
            Residence::Smem,
        ]));
    }

    /// The headline mix: one operand takes a shared stage while the other is read where it lies.
    /// The slot then rendezvouses for the staged one alone, which is what lets the pair share it.
    #[test]
    fn a_staged_operand_shares_a_slot_with_one_read_in_place() {
        let residences = [Residence::Smem, Residence::InPlace];
        assert!(compatible_slot_residences(&residences));
        let deliveries = rendezvous_deliveries(&residences, &[Delivery::Copy, Delivery::Copy]);
        assert_eq!(deliveries, vec![Delivery::Copy]);
    }

    /// Nothing materializes, so there is nothing to synchronize: the slot goes solo, which is what
    /// makes an all-in-place level equivalent to the plain recursive walk.
    #[test]
    fn an_all_in_place_slot_synchronizes_nothing() {
        let deliveries = rendezvous_deliveries(
            &[Residence::InPlace, Residence::InPlace],
            &[Delivery::Copy, Delivery::Procedural],
        );
        assert!(deliveries.is_empty());
        assert_eq!(slot_plan(&deliveries).0, Sync::Solo);
    }
}
