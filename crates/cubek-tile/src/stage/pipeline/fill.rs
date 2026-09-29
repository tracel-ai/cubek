//! Building a walk's slots and filling them over a [`Payload`].

use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use super::payload::base::{Payload, PayloadExpand, StageSpec};
use super::payload::pair::OperandPair;
use super::plan::StagePlan;
use crate::*;

#[cube]
impl<P: Payload<P> + Clone + CubeType<ExpandType: Clone>> Stages<P> {
    /// Build the `depth` slots staging `sources` into shared memory laid out as `storage`.
    ///
    /// `width` overrides the stage's line width; the operand must then be scalar and unpacked.
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
        let spec = comptime!(StageSpec {
            level: walk.level.clone(),
            depth: walk.depth(),
            storage,
            width,
        });

        // Later slots share buffers the first slot owns, so it is built first.
        let first = sources.staged(
            sources,
            comptime!(spec.clone()),
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
                    comptime!(spec.clone()),
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
    pub fn fill(&mut self, #[comptime] slot: usize, region: &Region) {
        let sources = self.sources.clone();
        self.slot_mut(slot)
            .refill(&sources, region, comptime!(Refill::EveryRegion));
    }
}

#[cube]
impl<P: Payload<P> + CubeType<ExpandType: Clone>> Slot<P> {
    /// Bring `src`'s window at `region` into this slot, for every operand whose refill is `only`.
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
impl<Lhs: Numeric, Rhs: Numeric> Stages<OperandPair<Lhs, Rhs>> {
    /// The `depth` slots staging both operands of a contraction into shared memory laid out as
    /// `storage`.
    pub fn smem(
        walk: &Walk,
        lhs: &Tile<Lhs>,
        rhs: &Tile<Rhs>,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
    ) -> Stages<OperandPair<Lhs, Rhs>> {
        let sources = OperandPair::<Lhs, Rhs> {
            lhs: lhs.clone(),
            rhs: rhs.clone(),
        };
        Stages::new(walk, &sources, storage, comptime!(None), depth)
    }
}

#[cube]
impl<T: Numeric> Stages<Tile<T>> {
    /// [`smem`](Stages::smem) for the sole operand `input`.
    pub fn smem_single(
        walk: &Walk,
        input: &Tile<T>,
        #[comptime] storage: StageStorage,
        #[comptime] depth: usize,
    ) -> Stages<Tile<T>> {
        Stages::new(walk, input, storage, comptime!(None), depth)
    }

    /// [`smem_single`](Stages::smem_single) with the stage served in `width`-wide lines.
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

// Line widths of a register-staged fill, bound by `fetch_buffers`.
define_size!(pub(crate) FL);
define_size!(pub(crate) FR);

/// Registers holding one unit's register-staged fill of both operands.
pub(crate) type FetchBuffers<Lhs, Rhs> = (Array<Vector<Lhs, FL>>, Array<Vector<Rhs, FR>>);

/// Bind the fetched line widths `FL` and `FR` for the rest of the kernel's scope.
#[cube]
fn register_fetched_widths(#[comptime] lhs: usize, #[comptime] rhs: usize) {
    intrinsic!(|scope| {
        scope.register_size::<FL>(lhs);
        scope.register_size::<FR>(rhs);
    });
}

#[cube]
impl<Lhs: Numeric, Rhs: Numeric> Stages<OperandPair<Lhs, Rhs>> {
    /// Registers holding one unit's fill of both operands across a contraction.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn fetch_buffers(&self) -> FetchBuffers<Lhs, Rhs> {
        let staged = self.slots.index(FIRST_SLOT);
        let lhs = staged.data.lhs.fetched_scalars();
        let rhs = staged.data.rhs.fetched_scalars();
        comptime!(assert!(
            lhs + rhs <= MOST_FETCHED_SCALARS,
            "Stages: a fill fetched into registers holds {lhs} + {rhs} scalars a unit, past the \
             {MOST_FETCHED_SCALARS} a unit keeps beside its contraction"
        ));
        let lhs_width = staged.data.lhs.vector_size();
        let rhs_width = staged.data.rhs.vector_size();
        register_fetched_widths(lhs_width, rhs_width);
        (
            Array::<Vector<Lhs, FL>>::new(comptime!(lhs / lhs_width)),
            Array::<Vector<Rhs, FR>>::new(comptime!(rhs / rhs_width)),
        )
    }

    /// Read this unit's share of filling slot `slot` for `region` into `lhs` and `rhs`.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn fetch(
        &self,
        #[comptime] slot: usize,
        region: &Region,
        lhs: &mut Array<Vector<Lhs, FL>>,
        rhs: &mut Array<Vector<Rhs, FR>>,
    ) {
        let staged = self.slots.index(slot);
        staged
            .data
            .lhs
            .fetch_from(&self.sources.lhs.at(region), lhs);
        staged
            .data
            .rhs
            .fetch_from(&self.sources.rhs.at(region), rhs);
    }

    /// Write what [`fetch`](Stages::fetch) read into slot `slot`; the caller owns the rendezvous.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn store(
        &mut self,
        #[comptime] slot: usize,
        lhs: &Array<Vector<Lhs, FL>>,
        rhs: &Array<Vector<Rhs, FR>>,
    ) {
        let slot = self.slot_mut(slot);
        slot.data.lhs.store_fetched(lhs);
        slot.data.rhs.store_fetched(rhs);
    }

    /// Refuse stages a register-staged schedule cannot drive.
    #[allow(dead_code)] // Reached through its expand, from `prefetched`.
    pub(crate) fn assert_copied_by_every_unit(&self) {
        let lhs = self.sources.lhs.delivery();
        let rhs = self.sources.rhs.delivery();
        comptime!(assert!(
            self.fillers == 0 && lhs == Delivery::Copy && rhs == Delivery::Copy,
            "Stages: a register-staged schedule fills its slots with every unit's own copy of \
             memory; these stages are filled by {} plane(s) of their own, or their sources are \
             delivered {lhs:?} and {rhs:?}",
            self.fillers
        ));
    }
}

// `consume` is spelled per payload shape: closure inference cannot resolve `P::ExpandType`.
impl<Lhs: Numeric, Rhs: Numeric> Slot<OperandPair<Lhs, Rhs>> {
    /// Wait for the slot's fill, pass its two tiles to `compute`, then free the slot.
    pub fn consume(&mut self, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> SlotExpand<OperandPair<Lhs, Rhs>> {
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
}

impl<T: Numeric> SlotExpand<Tile<T>> {
    pub fn __expand_consume_method<F>(&mut self, scope: &Scope, compute: F)
    where
        F: FnOnce(&Scope, &TileExpand<T>),
    {
        self.__expand_acquire_read_method(scope);
        compute(scope, &self.data);
        self.__expand_release_read_method(scope);
    }
}

impl<Lhs: Numeric, Rhs: Numeric> Stages<OperandPair<Lhs, Rhs>> {
    /// Consume slot `slot`: one step of a [`Compute`](Role::Compute) plane's walk.
    pub fn consume(&mut self, _slot: usize, _compute: impl FnOnce(&Tile<Lhs>, &Tile<Rhs>)) {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagesExpand<OperandPair<Lhs, Rhs>> {
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
