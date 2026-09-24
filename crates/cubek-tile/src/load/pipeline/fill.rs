//! Building a walk's slots and filling them: the one constructor ([`Stages::new`]) and the one
//! fill, both over a [`Payload`], plus the two entries a kernel spells its own schedule with.
//!
//! [`consume`](Slot::consume) is expanded by hand, per payload shape: it takes a closure so the
//! body stays the caller's, and closure-parameter inference cannot resolve `&T::ExpandType`
//! through a generic `T` the way it resolves a concrete `TileExpand`.

use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::payload::{Pair, PairExpand, Payload, PayloadExpand, Staging};
use super::plan::StagePlan;
use crate::*;

#[cube]
impl<P: Payload<P> + Clone + CubeType<ExpandType: Clone>> Stages<P> {
    /// Build the `depth` slots staging `sources` into shared memory laid out as `storage`, for a
    /// kernel walking `walk` itself, with [`Rendezvous`] deduced from the operands' delivery.
    ///
    /// Slot 0 allocates every buffer. A later slot allocates only what the walk refills: an
    /// operand whose window the walk leaves fixed is filled once, above the loop, and never
    /// rewritten, so one buffer serves the whole stages ([`Shared`](Refill::Shared)).
    ///
    /// `width` serves the stage in lines that wide rather than the operand's own: the buffer owns
    /// its layout, so an axis gmem cannot vectorize still reaches the leaf in lines. The operand
    /// must then be scalar and unquantized, with reads past its extent masked.
    // Reached through its expand, from the constructors below.
    #[allow(dead_code)]
    pub(crate) fn new(
        walk: &Walk,
        sources: &P,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] depth: usize,
    ) -> Stages<P> {
        let operands = sources.operands();
        let plan = comptime!(StagePlan::new(&operands, &walk.space, &walk.level));
        let staging = comptime!(Staging {
            level: walk.level.clone(),
            depth: walk.depth(),
            storage,
            width,
        });

        // The first slot owns every buffer, so it is built before the loop and handed to the
        // slots that share it. Nothing is shared at the first slot, so it stages against itself.
        let first = sources.staged(
            sources,
            comptime!(staging.clone()),
            comptime!(plan.refills(FIRST_SLOT)),
        );
        let mut slots = Sequence::<Slot<P>>::new();
        #[unroll]
        for slot in 0..depth {
            let data = if comptime!(slot == FIRST_SLOT) {
                first.clone()
            } else {
                sources.staged(
                    &first,
                    comptime!(staging.clone()),
                    comptime!(plan.refills(slot)),
                )
            };
            slots.push(Slot::wrap(
                data,
                Meeting::new(
                    comptime!(plan.sync()),
                    comptime!(plan.collective_full()),
                    comptime!(plan.fillers()),
                ),
                comptime!(plan.refills(slot)),
            ));
        }
        Stages::wrap(slots, sources.clone(), depth, comptime!(plan.fillers()))
    }

    /// Fill slot `slot` from `region`'s window: one step of a [`Fill`](Role::Fill) plane's walk.
    /// The stages knows its sources, so the caller names only which slot and which region.
    ///
    /// This is the same fill [`pipelined`](Stages::pipelined) runs one lap ahead of its own
    /// reads; here the two halves are separate walks, so the kernel spells the schedule and the
    /// stages spells the step.
    pub fn fill(&mut self, #[comptime] slot: usize, region: &Region) {
        let sources = self.sources.clone();
        self.slot_mut(slot)
            .refill(&sources, region, comptime!(Refill::EveryRegion));
    }
}

#[cube]
impl<P: Payload<P> + CubeType<ExpandType: Clone>> Slot<P> {
    /// Bring `src`'s window at `region` into this slot, for every operand whose refill is `only`.
    ///
    /// A streamed refill rendezvouses whatever it writes — the pipeline's phase belongs to the
    /// slot, not to any one operand — while a fixed one that has nothing to fill is skipped
    /// whole: it runs above the loop, where no reader is waiting on it.
    pub(crate) fn refill(&mut self, src: &P, region: &Region, #[comptime] only: Refill) {
        let refills = comptime!(self.refills.clone());
        let writes = comptime!(refills.contains(&only));
        if comptime!(writes || only == Refill::EveryRegion) {
            self.acquire_write();
            self.data
                .bring(src, &self.pipeline, region, comptime!(refills), only);
            self.release_write();
        }
    }
}

/// How the stages fill their own slots, at expand level: written once over a [`Payload`], since
/// none of it is the payload's shape.
impl<P: Payload<P> + CubeType<ExpandType: Clone>> StagesFill for StagesExpand<P> {
    fn has_fixed(&self, scope: &Scope) -> bool {
        self.slots
            .__expand_index_method(scope, FIRST_SLOT.into_expand(scope))
            .__expand_has_fixed_method(scope)
    }

    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let sources = self.sources.clone();
        self.__expand_slot_mut_method(scope, slot)
            .__expand_refill_method(scope, &sources, region, Refill::Once);
    }

    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand) {
        let sources = self.sources.clone();
        self.__expand_slot_mut_method(scope, slot)
            .__expand_refill_method(scope, &sources, region, Refill::EveryRegion);
    }

    fn publish(&mut self, scope: &Scope, slot: usize) {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_publish_method(scope);
    }
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Stages<Pair<Lhs, Rhs>> {
    /// [`new`](Stages::new) over the two operands of a contraction, which is what a walk stages
    /// where it stages more than one thing: the pair is built here rather than at every caller.
    pub fn smem(
        walk: &Walk,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
    ) -> Stages<Pair<Lhs, Rhs>> {
        let sources = Pair::<Lhs, Rhs> {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        };
        Stages::new(walk, &sources, storage, comptime!(None), depth)
    }
}

#[cube]
impl<T: Numeric> Stages<Tile<T>> {
    /// [`new`](Stages::new) for the sole operand `input`.
    pub fn smem_single(
        walk: &Walk,
        input: &Tile<T>,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
    ) -> Stages<Tile<T>> {
        Stages::new(walk, input, storage, comptime!(None), depth)
    }

    /// [`smem_single`](Stages::smem_single) with the stage served in `width`-wide lines rather
    /// than the operand's own.
    pub fn smem_single_at(
        walk: &Walk,
        input: &Tile<T>,
        #[comptime] storage: StageStorage,
        #[comptime] width: Option<usize>,
        #[comptime] depth: usize,
    ) -> Stages<Tile<T>> {
        Stages::new(walk, input, storage, width, depth)
    }
}

// `consume` takes a closure so the body stays caller-defined, which is why it is spelled per
// payload shape: inference resolves the pair's concrete `TileExpand` fields, not `P::ExpandType`.
impl<Lhs: Numeric, Rhs: Numeric> Slot<Pair<Lhs, Rhs>> {
    /// Consumer: wait the slot's fill, hand the two staged tiles to `compute`, then free the slot.
    /// A filled payload is already this region's; an in-place one is the whole operand, the caller
    /// selecting the region. See [`SlotExpand::__expand_consume_method`].
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }

    /// Producer: wait the slot is free, run `fill` over the staged buffers and the slot's
    /// [`Meeting`], then publish. What a fill *does* is the kernel's, which is why it is a
    /// closure. See [`SlotExpand::__expand_fill_method`].
    pub fn fill(&mut self, _fill: impl FnOnce(&mut Pair<Lhs, Rhs>, &Meeting)) {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> SlotExpand<Pair<Lhs, Rhs>> {
    pub fn __expand_fill_method<F>(&mut self, scope: &Scope, fill: F)
    where
        F: FnOnce(&Scope, &mut PairExpand<Lhs, Rhs>, &MeetingExpand),
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
        compute(scope, &self.data.lhs, &self.data.rhs);
        self.__expand_release_read_method(scope);
    }
}

impl<T: Numeric> Slot<Tile<T>> {
    /// [`consume`](Slot::consume) for the sole operand.
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<T>)) {
        unexpanded!()
    }

    /// [`fill`](Slot::fill) for the sole operand.
    pub fn fill(&mut self, _fill: impl FnOnce(&mut Tile<T>, &Meeting)) {
        unexpanded!()
    }
}

impl<T: Numeric> SlotExpand<Tile<T>> {
    pub fn __expand_fill_method<F>(&mut self, scope: &Scope, fill: F)
    where
        F: FnOnce(&Scope, &mut TileExpand<T>, &MeetingExpand),
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

impl<Lhs: Numeric, Rhs: Numeric> Stages<Pair<Lhs, Rhs>> {
    /// Consume slot `slot`: one step of a [`Compute`](Role::Compute) plane's walk. Waits the
    /// slot's fill, hands the staged tiles to `compute`, then frees the slot.
    pub fn consume(&mut self, _slot: usize, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagesExpand<Pair<Lhs, Rhs>> {
    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, slot: usize, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<Lhs>, &TileExpand<Rhs>),
    {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_consume_method(scope, compute);
    }
}

impl<T: Numeric> Stages<Tile<T>> {
    /// [`consume`](Stages::consume) for the sole operand.
    pub fn consume(&mut self, _slot: usize, _compute: impl FnOnce(&Tile<T>)) {
        unexpanded!()
    }
}

impl<T: Numeric> StagesExpand<Tile<T>> {
    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, slot: usize, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<T>),
    {
        self.__expand_slot_mut_method(scope, slot)
            .__expand_consume_method(scope, compute);
    }
}
