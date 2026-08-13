//! Filling a staging slot with matmul operands: the operand-deducing [`new`](Staging::new), the
//! pin split ([`fill_pinned`](Staging::fill_pinned) / [`fill_streamed`](Staging::fill_streamed)),
//! and the closure-driven [`fill`](Staging::fill) / [`consume`](Staging::consume) with their
//! hand-written expands.
//!
//! `fill`/`consume` are hand-written expand methods because a `Drop` guard can't emit a barrier
//! op in cubecl and `#[cube]` rejects `impl Trait` args.

use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

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
        // Staging copies an operand into a buffer shaped like its sub-tile. For a gathered operand
        // that is the *physical window* the sub-tile reads, compacted ([`Compaction`]), not its
        // logical box: several logical cells of a gather read the same physical one, so a logical
        // stage would replicate elements by roughly the tap count. The stage therefore keeps the
        // operand's own projection, and the gather stays where it already was, at the leaf's read
        // ([`MemData::smem_gathered`]).
        //
        // What a gathered stage does cost is gmem traffic: sibling windows overlap, so consecutive
        // regions re-read the halo between them. And the window is only smaller than the logical
        // box when the taps outrun the stride; past that it is padding the fill still reads
        // ([`Compaction`]).
        let lhs_gathered = lhs.gathered();
        let rhs_gathered = rhs.gathered();
        let gathered = comptime!(lhs_gathered || rhs_gathered);
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
        // How the operands stage follows from what they become; they agree or meet the
        // kind-pairing panics at the instruction.
        let stage = comptime!(out.operand_stage(lhs.leaf));
        match comptime!(stage) {
            OperandStage::Plane => {
                comptime!(assert!(
                    !lhs_delivery.is_tma() && !rhs_delivery.is_tma(),
                    "Staging: a TMA source cannot stage into plane tiles"
                ));
                // A fragment fill reads its source as a 2-D window, which a gathered operand has
                // no equivalent of. `OperandStage::Smem` stages the physical window instead and
                // leaves the gather to the leaf's read; a cmma leaf (`OperandStage::Plane`) has no
                // such path, so a gathered operand cannot feed one today.
                comptime!(assert!(
                    !gathered,
                    "Staging: a gathered operand cannot stage into plane tiles (OperandStage::Plane); \
                     only OperandStage::Smem stages one, as the compacted window its leaf reads"
                ));
                // Each operand's fragment encoding is its own; `out` only names the contracted axis.
                let a = PlanePartition::store(
                    comptime!(lhs.space.divide()),
                    comptime!(lhs.leaf),
                    comptime!(out.clone()),
                );
                let b = PlanePartition::store(
                    comptime!(rhs.space.divide()),
                    comptime!(rhs.leaf),
                    comptime!(out.clone()),
                );
                Staging::wrap((a, b), Pipeline::new(Sync::Solo), pin_lhs, pin_rhs, stage)
            }
            OperandStage::Smem => {
                let sync = comptime!(Sync::of(lhs_delivery, rhs_delivery));
                // Each operand carries how far its quantized form travels, so staging it asks
                // nothing: `smem_like` allocates the element that operand needs staged.
                let stages = (MemData::smem_like(lhs), MemData::smem_like(rhs));
                Staging::wrap(stages, Pipeline::new(sync), pin_lhs, pin_rhs, stage)
            }
        }
    }

    /// Fill the pinned operand(s), those the walk leaves invariant, from `region`'s window.
    /// Their window never moves, so `region` is region 0 and this runs once, above the loop.
    /// A no-op when nothing is pinned (both operands stream).
    pub fn fill_pinned(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let pin_lhs = comptime!(self.pin_lhs);
        let pin_rhs = comptime!(self.pin_rhs);
        if comptime!(pin_lhs || pin_rhs) {
            self.fill(|staged_operands, pipe| {
                if comptime!(pin_lhs) {
                    pipe.fill(&mut staged_operands.0, &lhs.at(region));
                }
                if comptime!(pin_rhs) {
                    pipe.fill(&mut staged_operands.1, &rhs.at(region));
                }
            });
        }
    }

    /// Fill the streamed operand(s), everything not pinned, from `region`'s window. Runs per
    /// region inside the walk.
    pub fn fill_streamed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let pin_lhs = comptime!(self.pin_lhs);
        let pin_rhs = comptime!(self.pin_rhs);
        self.fill(|staged_operands, pipe| {
            if comptime!(!pin_lhs) {
                pipe.fill(&mut staged_operands.0, &lhs.at(region));
            }
            if comptime!(!pin_rhs) {
                pipe.fill(&mut staged_operands.1, &rhs.at(region));
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
