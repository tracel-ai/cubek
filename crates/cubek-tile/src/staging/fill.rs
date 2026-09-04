//! Building and filling the staging slots of a ring: the [`SlotPlan`] every slot shares, the
//! shared-memory ring constructors ([`Ring::smem`] / [`Ring::smem_single`]), the fixed/streamed
//! split ([`fill_fixed`](Staging::fill_fixed) / [`fill_streamed`](Staging::fill_streamed)), and
//! the closure-driven [`fill`](Staging::fill) / [`consume`](Staging::consume) with their
//! hand-written expands.
//!
//! `fill`/`consume` are hand-written expand methods because a `Drop` guard can't emit a barrier
//! op in cubecl and `#[cube]` rejects `impl Trait` args.

use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;
use cubecl::zspace::SmallVec;

use crate::*;

/// One operand's comptime facts, read off its [`Tile`] before any slot is built.
pub(crate) struct SlotOperand<'a> {
    delivery: Delivery,
    space: &'a Space,
}

impl<'a> SlotOperand<'a> {
    pub(crate) fn new(delivery: Delivery, space: &'a Space) -> Self {
        SlotOperand { delivery, space }
    }
}

/// What every slot of one ring decides at comptime: how it rendezvouses, and per operand when it
/// is filled. Deduced once from the operands, so a ring's slots agree by construction rather than
/// by each re-deriving the same answers off the same tiles.
#[derive(Clone, PartialEq, Debug)]
pub(crate) struct SlotPlan {
    operands: Vec<OperandPlan>,
    sync: Sync,
    collective_full: bool,
}

impl SlotPlan {
    pub(crate) fn new(operands: &[SlotOperand], op_space: &Space, level: &Level) -> SlotPlan {
        // Fix an operand only when its window is genuinely invariant across the walk. A barrier
        // pipeline arrives `full` once per fill, so a TMA pair keeps the joint per-region fill;
        // splitting an invariant out would corrupt its phase. A dynamic level can't decide
        // invariance at comptime. Both fall back to streaming.
        let can_fix_invariants =
            op_space.is_static() && !operands.iter().any(|op| op.delivery.is_tma());
        let planned_operands = operands
            .iter()
            .map(|op| {
                let mode = if can_fix_invariants && level.walk_invariant(op_space, op.space) {
                    WindowMode::Fixed
                } else {
                    WindowMode::Streamed
                };
                OperandPlan {
                    mode,
                    delivery: op.delivery,
                }
            })
            .collect();
        let deliveries: Vec<_> = operands.iter().map(|op| op.delivery).collect();
        SlotPlan {
            operands: planned_operands,
            sync: Sync::for_deliveries(&deliveries),
            collective_full: Sync::collective_full(&deliveries),
        }
    }

    /// What ring slot `slot` holds for operand `operand`. The first slot owns every buffer; a
    /// later slot reuses the first slot's for whatever the walk never rewrites.
    pub(crate) fn operand_plan(&self, operand: usize, slot: usize) -> OperandPlan {
        let mut plan = self.operands[operand];
        if slot != FIRST_SLOT {
            plan.mode = plan.mode.in_later_slot();
        }
        plan
    }

    /// Whether slot `slot` takes the first slot's buffer for `operand` instead of allocating one.
    pub(crate) fn reuses_first_buffer(&self, operand: usize, slot: usize) -> bool {
        self.operand_plan(operand, slot).mode == WindowMode::Reused
    }

    pub(crate) fn sync(&self) -> Sync {
        self.sync
    }

    pub(crate) fn collective_full(&self) -> bool {
        self.collective_full
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Ring<(Tile<Lhs>, Tile<Rhs>)> {
    /// Build the `depth` slots staging both operands into shared memory laid out as `storage`,
    /// for a kernel walking `walk` itself, with [`Sync`] deduced from the operands' delivery.
    ///
    /// Slot 0 allocates every buffer. A later slot allocates only what the walk refills: an
    /// operand whose window the walk leaves fixed is filled once, above the loop, and never
    /// rewritten, so one buffer serves the whole ring ([`Reused`](WindowMode::Reused)).
    pub fn smem(
        walk: &Walk,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
    ) -> Ring<(Tile<Lhs>, Tile<Rhs>)> {
        let lhs_delivery = lhs.delivery();
        let rhs_delivery = rhs.delivery();
        let plan = comptime!(SlotPlan::new(
            &[
                SlotOperand::new(lhs_delivery, &lhs.space),
                SlotOperand::new(rhs_delivery, &rhs.space),
            ],
            &walk.space,
            &walk.level,
        ));
        let mut slots = Sequence::<Staging<(Tile<Lhs>, Tile<Rhs>)>>::new();
        #[unroll]
        for slot in 0..depth {
            let staged_lhs = if comptime!(plan.reuses_first_buffer(FIRST, slot)) {
                slots.index(FIRST_SLOT).data.0.clone()
            } else {
                stage_smem(lhs, storage, comptime!(None))
            };
            let staged_rhs = if comptime!(plan.reuses_first_buffer(SECOND, slot)) {
                slots.index(FIRST_SLOT).data.1.clone()
            } else {
                stage_smem(rhs, storage, comptime!(None))
            };
            let staging = Staging::wrap(
                (staged_lhs, staged_rhs),
                Pipeline::new(comptime!(plan.sync()), comptime!(plan.collective_full())),
                comptime!(SmallVec::from_slice(&[
                    plan.operand_plan(FIRST, slot),
                    plan.operand_plan(SECOND, slot),
                ])),
            );
            slots.push(staging);
        }
        Ring::wrap(slots, (lhs.clone(), rhs.clone()), depth)
    }
}

impl<Lhs: Numeric, Rhs: Numeric> RingFill for RingExpand<(Tile<Lhs>, Tile<Rhs>)> {
    fn has_fixed(&self, scope: &Scope) -> bool {
        self.slots
            .__expand_index_method(scope, FIRST_SLOT.into_expand(scope))
            .__expand_has_fixed_method(scope)
    }

    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let (lhs, rhs) = (self.sources.0.clone(), self.sources.1.clone());
        self.__expand_slot_mut_method(scope, slot)
            .__expand_fill_fixed_method(scope, &lhs, &rhs, region);
    }

    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let (lhs, rhs) = (self.sources.0.clone(), self.sources.1.clone());
        self.__expand_slot_mut_method(scope, slot)
            .__expand_fill_streamed_method(scope, &lhs, &rhs, region);
    }

    fn publish(&mut self, scope: &Scope, slot: usize) {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_publish_method(scope);
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Fill the fixed operand(s), those the walk leaves invariant, from `region`'s window.
    /// Their window never moves, so `region` is region 0 and this runs once, above the loop.
    /// A no-op when nothing is fixed, and when a later ring slot reuses the first's buffers.
    pub(crate) fn fill_fixed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let lhs_plan = self.plan(FIRST);
        let rhs_plan = self.plan(SECOND);
        let fixed_lhs = comptime!(lhs_plan.mode == WindowMode::Fixed);
        let fixed_rhs = comptime!(rhs_plan.mode == WindowMode::Fixed);
        if comptime!(fixed_lhs || fixed_rhs) {
            self.fill(|staged_operands, pipe| {
                if comptime!(fixed_lhs) {
                    pipe.fill(&mut staged_operands.0, &lhs.at(region));
                }
                if comptime!(fixed_rhs) {
                    pipe.fill(&mut staged_operands.1, &rhs.at(region));
                }
            });
        }
    }

    /// Fill the streamed operand(s), everything the walk moves, from `region`'s window. Runs per
    /// region inside the walk. The slot still rendezvouses when every operand is fixed: the
    /// pipeline's phase belongs to the slot, not to any one operand.
    pub fn fill_streamed(&mut self, lhs: &Tile<Lhs>, rhs: &Tile<Rhs>, region: &Region) {
        let lhs_plan = self.plan(FIRST);
        let rhs_plan = self.plan(SECOND);
        let stream_lhs = comptime!(lhs_plan.mode == WindowMode::Streamed);
        let stream_rhs = comptime!(rhs_plan.mode == WindowMode::Streamed);
        self.fill(|staged_operands, pipe| {
            if comptime!(stream_lhs) {
                pipe.fill(&mut staged_operands.0, &lhs.at(region));
            }
            if comptime!(stream_rhs) {
                pipe.fill(&mut staged_operands.1, &rhs.at(region));
            }
        });
    }
}

// `fill`/`consume` take closures so the body stays caller-defined (fill each buffer however, run the
// mma). They're provided for the `(Tile<Lhs>, Tile<Rhs>)` payload (not generic `T`): closure-parameter
// inference can't resolve the projection `&mut T::ExpandType` through a generic `T`, but resolves the
// concrete `TileExpand` fields of the pair directly.
impl<Lhs: Numeric, Rhs: Numeric> Staging<(Tile<Lhs>, Tile<Rhs>)> {
    /// Producer: wait the slot is free, run `fill` over the staged buffers and the slot's
    /// [`Pipeline`], then publish. See [`StagingExpand::__expand_fill_method`].
    pub fn fill(&mut self, _fill: impl FnOnce(&mut (Tile<Lhs>, Tile<Rhs>), &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the two staged tiles to `compute`, then free the slot.
    /// Each tile's bytes and runtime map were stored together by the producer. A payload the slot
    /// filled is already this region's; an in-place one is the operand whole, and the caller
    /// selects the region out of it ([`read_operand`]).
    /// See [`StagingExpand::__expand_consume_method`].
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
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
}

#[cube]
impl<T: Numeric> Ring<Tile<T>> {
    /// [`smem`](Ring::smem) for the sole operand `input`.
    pub fn smem_single(
        walk: &Walk,
        input: &Tile<T>,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
    ) -> Ring<Tile<T>> {
        Ring::<Tile<T>>::smem_single_at(walk, input, storage, comptime!(None), depth)
    }

    /// [`smem_single`](Ring::smem_single) with the stage served in `width`-wide lines rather
    /// than the operand's own: the buffer owns its layout, so an axis global memory could not
    /// vectorize still reaches the leaf in lines. The operand must be scalar and unquantized,
    /// and its reads past the real extent masked, which is the launch's to state.
    pub fn smem_single_at(
        walk: &Walk,
        input: &Tile<T>,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] depth: usize,
    ) -> Ring<Tile<T>> {
        let delivery = input.delivery();
        let plan = comptime!(SlotPlan::new(
            &[SlotOperand::new(delivery, &input.space)],
            &walk.space,
            &walk.level,
        ));

        let mut slots = Sequence::<Staging<Tile<T>>>::new();
        #[unroll]
        for slot in 0..depth {
            let staged_input = if comptime!(plan.reuses_first_buffer(FIRST, slot)) {
                slots.index(FIRST_SLOT).data.clone()
            } else {
                stage_smem(input, storage, width)
            };
            let staging = Staging::wrap(
                staged_input,
                Pipeline::new(comptime!(plan.sync()), comptime!(plan.collective_full())),
                comptime!(SmallVec::from_slice(&[plan.operand_plan(FIRST, slot)])),
            );
            slots.push(staging);
        }
        Ring::wrap(slots, input.clone(), depth)
    }
}

impl<T: Numeric> RingFill for RingExpand<Tile<T>> {
    fn has_fixed(&self, scope: &Scope) -> bool {
        self.slots
            .__expand_index_method(scope, FIRST_SLOT.into_expand(scope))
            .__expand_has_fixed_method(scope)
    }

    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let input = self.sources.clone();
        self.__expand_slot_mut_method(scope, slot)
            .__expand_fill_fixed_method(scope, &input, region);
    }

    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let input = self.sources.clone();
        self.__expand_slot_mut_method(scope, slot)
            .__expand_fill_streamed_method(scope, &input, region);
    }

    fn publish(&mut self, scope: &Scope, slot: usize) {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_publish_method(scope);
    }
}

#[cube]
impl<T: Numeric> Staging<Tile<T>> {
    /// Fill the operand from `region`'s window if the walk leaves it fixed.
    pub(crate) fn fill_fixed(&mut self, input: &Tile<T>, region: &Region) {
        let plan = self.plan(FIRST);
        let fixed = comptime!(plan.mode == WindowMode::Fixed);
        if comptime!(fixed) {
            self.fill(|s, pipe| {
                pipe.fill(s, &input.at(region));
            });
        }
    }

    /// Fill the operand from `region`'s window if the walk moves it. The slot still rendezvouses
    /// otherwise: the pipeline's phase belongs to the slot, not to the operand.
    pub fn fill_streamed(&mut self, input: &Tile<T>, region: &Region) {
        let plan = self.plan(FIRST);
        let stream = comptime!(plan.mode == WindowMode::Streamed);
        self.fill(|s, pipe| {
            if comptime!(stream) {
                pipe.fill(s, &input.at(region));
            }
        });
    }
}

impl<T: Numeric> Staging<Tile<T>> {
    /// Producer: wait the slot is free, run `fill` over the staged buffer and the slot's
    /// [`Pipeline`], then publish.
    pub fn fill(&mut self, _fill: impl FnOnce(&mut Tile<T>, &Pipeline)) {
        unexpanded!()
    }

    /// Consumer: wait the slot's fill, hand the staged tile to `compute`, then free the slot.
    /// Each tile's bytes and runtime map were stored together by the producer. A payload the slot
    /// filled is already this region's; an in-place one is the operand whole, and the caller
    /// selects the region out of it ([`read_operand`]).
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<T>)) {
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
}
/// Allocate one shared-memory stage for `input` laid out as `storage`, served at `width` where
/// one is stated ([`Ring::smem_single_at`]). A gathered operand keeps its compacted physical
/// window and projection, so staging does not replicate each logical element for every gather
/// tap; the leaf performs the gather on read instead.
#[cube]
fn stage_smem<T: Numeric>(
    input: &Tile<T>,
    #[comptime] storage: StageStorage,
    #[comptime] width: Option<usize>,
) -> Tile<T> {
    MemData::stage(input, storage, width)
}

#[cfg(test)]
mod tests {
    use super::*;

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    /// A one-level space over `M`/`N`/`K`, plus a projection per operand so a slot can be planned
    /// against it. `lhs` spans `M`/`K`, `rhs` spans `K`/`N`, so a `K` walk moves both.
    fn spaces() -> (Space, Space, Space) {
        let space = Tiling::over(&[(M, 8), (N, 8), (K, 8)])
            .level(|l| {
                l.walk(&[(M, 8), (N, 8), (K, 4)]);
            })
            .build();
        let lhs = space.project(&[M, K]);
        let rhs = space.project(&[K, N]);
        (space, lhs, rhs)
    }

    fn operand(delivery: Delivery, space: &Space) -> SlotOperand<'_> {
        SlotOperand::new(delivery, space)
    }

    #[test]
    fn a_streamed_operand_is_rebuilt_in_every_slot() {
        let (space, lhs, rhs) = spaces();
        let plan = SlotPlan::new(
            &[operand(Delivery::Copy, &lhs), operand(Delivery::Copy, &rhs)],
            &space,
            space.partitioner().level(),
        );
        for slot in 0..2 {
            assert_eq!(plan.operand_plan(FIRST, slot).mode, WindowMode::Streamed);
            assert!(!plan.reuses_first_buffer(FIRST, slot));
        }
    }

    /// An operand whose window the walk never moves is filled once and shares its buffer.
    #[test]
    fn a_fixed_operand_reuses_the_first_slots_buffer() {
        let space = Tiling::over(&[(M, 8), (N, 8), (K, 8)])
            .level(|l| {
                l.walk(&[(M, 8), (N, 4), (K, 8)]);
            })
            .build();
        let lhs = space.project(&[M, K]);
        let rhs = space.project(&[K, N]);
        let plan = SlotPlan::new(
            &[operand(Delivery::Copy, &lhs), operand(Delivery::Copy, &rhs)],
            &space,
            space.partitioner().level(),
        );
        assert_eq!(plan.operand_plan(FIRST, 0).mode, WindowMode::Fixed);
        assert_eq!(plan.operand_plan(FIRST, 1).mode, WindowMode::Reused);
        assert!(plan.reuses_first_buffer(FIRST, 1));
        assert_eq!(plan.operand_plan(SECOND, 1).mode, WindowMode::Streamed);
    }

    /// A barrier pipeline arrives once per fill, so a TMA operand streams even when fixed.
    #[test]
    fn a_tma_operand_is_never_fixed() {
        let space = Tiling::over(&[(M, 8), (N, 8), (K, 8)])
            .level(|l| {
                l.walk(&[(M, 8), (N, 4), (K, 8)]);
            })
            .build();
        let lhs = space.project(&[M, K]);
        let rhs = space.project(&[K, N]);
        let plan = SlotPlan::new(
            &[operand(Delivery::Tma, &lhs), operand(Delivery::Tma, &rhs)],
            &space,
            space.partitioner().level(),
        );
        assert_eq!(plan.operand_plan(FIRST, 0).mode, WindowMode::Streamed);
    }
}
