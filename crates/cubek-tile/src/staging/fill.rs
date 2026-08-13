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
        // Both operands use the output leaf's staging kind. The helper preserves each tile's
        // own projection and rejects unsupported Plane/TMA/gather combinations.
        let stage = comptime!(out.operand_stage(lhs.leaf));
        let stages = (
            stage_operand(lhs, comptime!(out.clone()), stage),
            stage_operand(rhs, comptime!(out.clone()), stage),
        );
        let sync = match comptime!(stage) {
            OperandStage::Plane => comptime!(Sync::Solo),
            OperandStage::Smem => comptime!(Sync::merge(lhs_delivery, rhs_delivery)),
        };
        Staging::wrap(stages, Pipeline::new(sync), pin_lhs, pin_rhs, stage)
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
        let stage = comptime!(out.operand_stage(input.leaf));
        let staged_input = stage_operand(input, comptime!(out.clone()), stage);
        let sync = match comptime!(stage) {
            OperandStage::Plane => comptime!(Sync::Solo),
            OperandStage::Smem => comptime!(Sync::from(delivery)),
        };
        Staging::wrap(staged_input, Pipeline::new(sync), pin, false, stage)
    }

    /// Fill the pinned operand from `region`'s window.
    pub fn fill_pinned(&mut self, input: &Tile<T>, region: &Region) {
        let pin = self.pinned();
        if comptime!(pin) {
            self.fill(|s, pipe| {
                pipe.fill(s, &input.at(region));
            });
        }
    }

    /// Fill the streamed operand from `region`'s window.
    pub fn fill_streamed(&mut self, input: &Tile<T>, region: &Region) {
        let pin = self.pinned();
        self.fill(|s, pipe| {
            if comptime!(!pin) {
                pipe.fill(s, &input.at(region));
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
