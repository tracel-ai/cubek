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

/// The slot-wide rendezvous derived only from operands that physically materialize.
#[derive(Clone, Copy)]
struct SlotPlan {
    sync: Sync,
    collective_full: bool,
}

fn slot_plan(deliveries: &[Delivery]) -> SlotPlan {
    let sync = if deliveries.is_empty() {
        Sync::Solo
    } else {
        Sync::for_deliveries(deliveries)
    };
    SlotPlan {
        sync,
        collective_full: Sync::collective_full(deliveries),
    }
}

fn rendezvous_deliveries(stages: &[OperandStage], deliveries: &[Delivery]) -> Vec<Delivery> {
    stages
        .iter()
        .zip(deliveries)
        .filter_map(|(stage, delivery)| (*stage == OperandStage::Smem).then_some(*delivery))
        .collect()
}

/// A plane partition is private to its unit and assumes a solo fill. It cannot share a slot with
/// a shared-memory stage, whose transfer selects a slot-wide Cube or Barrier pipeline.
fn compatible_slot_stages(stages: &[OperandStage]) -> bool {
    let has_plane = stages.iter().any(|stage| *stage == OperandStage::Plane);
    let has_smem = stages.iter().any(|stage| *stage == OperandStage::Smem);
    !(has_plane && has_smem)
}

/// Fill a materialized operand through its slot pipeline, or rebind an in-place procedural
/// operand to the source region. The pipeline is slot-wide while this decision is per-operand.
#[cube]
fn fill_operand<T: Numeric>(
    dst: &mut Tile<T>,
    src: &Tile<T>,
    #[comptime] stage: OperandStage,
    pipe: &Pipeline,
) {
    if comptime!(stage == OperandStage::InPlace) {
        // In-place procedural tiles have no physical fill. `copy_from` rebinds their runtime
        // origin to this region without involving the slot pipeline.
        dst.copy_from(src);
    } else {
        pipe.fill(dst, src);
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Build a slot staging one region of the operands `lhs`/`rhs`. An [`OperandStage::Plane`]
    /// stages into plane-private tile partitions ([`Solo`](Sync::Solo)); [`Smem`](OperandStage::Smem)
    /// takes fresh shared memory, with [`Sync`] deduced from the operands' delivery.
    pub fn new(
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
    ) -> Staging<(Tile<Lhs>, Tile<Rhs>)> {
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
        let lhs_plan = lhs.stage();
        let rhs_plan = rhs.stage();
        let lhs_requested = comptime!(out.operand_stage(lhs.leaf));
        let rhs_requested = comptime!(out.operand_stage(rhs.leaf));
        let lhs_stage = comptime!(lhs_plan.resolve(lhs.leaf, lhs_requested));
        let rhs_stage = comptime!(rhs_plan.resolve(rhs.leaf, rhs_requested));
        comptime!(assert!(
            compatible_slot_stages(&[lhs_stage, rhs_stage]),
            "Staging: Plane and Smem operands cannot share a slot"
        ));
        let materialized = comptime!(rendezvous_deliveries(
            &[lhs_stage, rhs_stage],
            &[lhs_delivery, rhs_delivery],
        ));
        let plan = comptime!(slot_plan(&materialized));
        let stages = (
            stage_operand(lhs, comptime!(out.clone()), lhs_stage),
            stage_operand(rhs, comptime!(out.clone()), rhs_stage),
        );
        Staging::wrap(
            stages,
            Pipeline::new(comptime!(plan.sync), comptime!(plan.collective_full)),
            pin_lhs,
            pin_rhs,
            lhs_stage,
            comptime!(Option::Some(rhs_stage)),
        )
    }

    /// Fill the pinned operand(s), those the walk leaves invariant, from `region`'s window.
    /// Their window never moves, so `region` is region 0 and this runs once, above the loop.
    /// A no-op when nothing is pinned (both operands stream).
    pub fn fill_pinned(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let pin_lhs = comptime!(self.pin_lhs);
        let pin_rhs = comptime!(self.pin_rhs);
        let lhs_stage = comptime!(self.stage_lhs);
        let rhs_stage = comptime!(match self.stage_rhs {
            Option::Some(stage) => stage,
            Option::None => panic!("binary staging slot has an rhs stage"),
        });
        if comptime!(pin_lhs || pin_rhs) {
            self.fill(|staged_operands, pipe| {
                if comptime!(pin_lhs) {
                    let source = lhs.at(region);
                    fill_operand(&mut staged_operands.0, &source, lhs_stage, pipe);
                }
                if comptime!(pin_rhs) {
                    let source = rhs.at(region);
                    fill_operand(&mut staged_operands.1, &source, rhs_stage, pipe);
                }
            });
        }
    }

    /// Fill the streamed operand(s), everything not pinned, from `region`'s window. Runs per
    /// region inside the walk.
    pub fn fill_streamed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let pin_lhs = comptime!(self.pin_lhs);
        let pin_rhs = comptime!(self.pin_rhs);
        let lhs_stage = comptime!(self.stage_lhs);
        let rhs_stage = comptime!(match self.stage_rhs {
            Option::Some(stage) => stage,
            Option::None => panic!("binary staging slot has an rhs stage"),
        });
        self.fill(|staged_operands, pipe| {
            if comptime!(!pin_lhs) {
                let source = lhs.at(region);
                fill_operand(&mut staged_operands.0, &source, lhs_stage, pipe);
            }
            if comptime!(!pin_rhs) {
                let source = rhs.at(region);
                fill_operand(&mut staged_operands.1, &source, rhs_stage, pipe);
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
        let delivery = input.delivery();
        // Pin only when the window is genuinely fixed across the walk: a TMA pair keeps the joint
        // per-region fill (its barrier pipeline arrives `full` once per fill), and a dynamic level
        // can't decide invariance at comptime. Both fall back to streaming (pin = false).
        let split = comptime!(op_space.is_static() && !delivery.is_tma());
        let pin = comptime!(split && op_space.walk_invariant(&input.space));
        let input_plan = input.stage();
        let requested = comptime!(out.operand_stage(input.leaf));
        let stage = comptime!(input_plan.resolve(input.leaf, requested));
        let materialized = comptime!(rendezvous_deliveries(&[stage], &[delivery]));
        let plan = comptime!(slot_plan(&materialized));
        let staged_input = stage_operand(input, comptime!(out.clone()), stage);
        Staging::wrap(
            staged_input,
            Pipeline::new(comptime!(plan.sync), comptime!(plan.collective_full)),
            pin,
            false,
            stage,
            comptime!(Option::None),
        )
    }

    /// Fill the pinned operand from `region`'s window.
    pub fn fill_pinned(&mut self, input: &Tile<T>, region: &Region) {
        let pin = self.pinned();
        let stage = self.stage();
        if comptime!(pin) {
            self.fill(|s, pipe| {
                let source = input.at(region);
                fill_operand(s, &source, stage, pipe);
            });
        }
    }

    /// Fill the streamed operand from `region`'s window.
    pub fn fill_streamed(&mut self, input: &Tile<T>, region: &Region) {
        let pin = self.pinned();
        let stage = self.stage();
        self.fill(|s, pipe| {
            if comptime!(!pin) {
                let source = input.at(region);
                fill_operand(s, &source, stage, pipe);
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

/// Allocate one staged operand for `stage`. A gathered operand keeps its compacted physical
/// window and projection, so staging does not replicate each logical element for every gather
/// tap. The leaf performs the gather on read instead; that keeps staging compact but means
/// adjacent logical regions may re-read overlapping halo cells.
#[cube]
fn stage_operand<T: Numeric>(
    input: &Tile<T>,
    #[comptime] out: Space,
    #[comptime] stage: OperandStage,
) -> Tile<T> {
    let gathered = input.gathered();
    let delivery = input.delivery();
    match comptime!(stage) {
        OperandStage::InPlace => match &input.tile_kind {
            TileKind::Procedural(data) => Tile::<T> {
                tile_kind: TileKind::new_Procedural(data.clone()),
                space: comptime!(input.space.divide()),
                leaf: comptime!(input.leaf),
            },
            TileKind::Gmem(_)
            | TileKind::Smem(_)
            | TileKind::PlaneTile(_)
            | TileKind::PlanePartition(_)
            | TileKind::TmaGmem(_) => {
                panic!("Staging: only procedural operands currently execute in place")
            }
        },
        OperandStage::Plane => {
            comptime!(assert!(
                !delivery.is_tma(),
                "Staging: a TMA source cannot stage into plane tiles"
            ));
            comptime!(assert!(
                !gathered,
                "Staging: a gathered operand cannot stage into plane tiles (OperandStage::Plane); \
                 only OperandStage::Smem stages one, as the compacted window its leaf reads"
            ));
            PlanePartition::store(
                comptime!(input.space.divide()),
                comptime!(input.leaf),
                comptime!(out.clone()),
            )
        }
        OperandStage::Smem => MemData::smem_like(input),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn an_in_place_operand_does_not_join_the_slot_rendezvous() {
        let deliveries = rendezvous_deliveries(
            &[OperandStage::InPlace, OperandStage::Smem],
            &[Delivery::Procedural, Delivery::Copy],
        );
        assert_eq!(deliveries, vec![Delivery::Copy]);
        assert_eq!(slot_plan(&deliveries).sync, Sync::Cube);
    }

    #[test]
    fn a_plane_stage_cannot_share_a_slot_with_smem() {
        assert!(!compatible_slot_stages(&[
            OperandStage::Plane,
            OperandStage::Smem,
        ]));
        assert!(!compatible_slot_stages(&[
            OperandStage::Smem,
            OperandStage::Plane,
        ]));
        assert!(compatible_slot_stages(&[
            OperandStage::Plane,
            OperandStage::InPlace,
        ]));
    }

    #[test]
    fn materialization_resolves_per_operand() {
        assert_eq!(
            StagePlan::in_place().resolve(Leaf::Memory, OperandStage::Smem),
            OperandStage::InPlace,
        );
        assert_eq!(
            StagePlan::for_leaf(Leaf::Memory).resolve(Leaf::Memory, OperandStage::Smem),
            OperandStage::Smem,
        );
    }

    #[test]
    fn a_fragment_consumer_forces_procedural_smem() {
        assert_eq!(
            StagePlan::in_place().resolve(Leaf::Cmma, OperandStage::Plane),
            OperandStage::Smem,
        );
    }
}
