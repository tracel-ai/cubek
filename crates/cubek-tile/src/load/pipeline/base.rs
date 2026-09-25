//! The one buffered walk: `depth` [`Slot`] slots driven as a circular software pipeline, so
//! `depth - 1` regions are in flight while one computes.
//!
//! Single and double buffering are this schedule at `depth` 1 and 2. At depth 1 a region's fill is
//! the last event before its read, so the consume publishes it; deeper, the fill a lap ahead does.
//! That, prologue and drain are the protocol ([`pipelined`](Stages::pipelined)); a kernel
//! supplies the stages and the consume.
//!
//! [`prefetched`](Stages::prefetched) is the same walk with each region's fill split around the
//! contraction before it, the next region's loads held in the units' registers meanwhile.

use cubecl::frontend::branch::{if_else_expand, if_expand};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// What a plane of the cube does with the stages' slots ([`Stages::role`]).
#[derive(CubeType, CubeTypeMut, IntoRuntime)]
#[cube(runtime_variants)]
pub enum Role {
    /// Fills the slots, and takes no tile of any level ([`Levels::filled_by`](crate::Levels::filled_by)).
    Fill,
    /// Reads the slots and computes out of them, and fills none.
    Compute,
}

/// The `depth` slots of one buffered walk, and the operands they are filled from: the same
/// payload shape at this level, so [`pipelined`](Stages::pipelined) can fill a slot for a region on its own.
#[derive(CubeType)]
pub struct Stages<T: CubeType> {
    pub(crate) slots: Sequence<Slot<T>>,
    pub(crate) sources: T,
    // Read by the two schedules ([`pipelined`](Stages::pipelined),
    // [`prefetched`](Stages::prefetched)) at expand level only.
    #[allow(dead_code)]
    #[cube(comptime)]
    pub(crate) depth: usize,
    /// Planes of the cube that fill these slots and do nothing else ([`Levels::filled_by`](crate::Levels::filled_by)).
    #[cube(comptime)]
    pub(crate) fillers: usize,
}

#[cube]
impl<T: CubeType> Stages<T> {
    /// Wrap already-built slots over the operands they stage. The public entries are
    /// [`smem`](Stages::smem) and [`smem_single`](Stages::smem_single).
    pub(crate) fn wrap(
        slots: Sequence<Slot<T>>,
        sources: T,
        #[comptime] depth: usize,
        #[comptime] fillers: usize,
    ) -> Stages<T> {
        comptime!(assert!(
            depth > 0,
            "Stages: a pipeline needs at least one slot"
        ));
        Stages::<T> {
            slots,
            sources,
            depth,
            fillers,
        }
    }

    /// What this plane does with the stages' slots. The planes a walk sets aside to fill sit at
    /// the end of the cube ([`Levels::filled_by`](crate::Levels::filled_by)), so a unit fills exactly when it stands at or
    /// past the ones that compute, and every plane below is the one it would have been.
    ///
    /// Where the walk set none aside, every plane computes and fills its own slots, which is the
    /// schedule [`pipelined`](Stages::pipelined) writes.
    pub fn role(&self) -> Role {
        if comptime!(self.fillers == 0) {
            Role::new_Compute()
        } else if UNIT_POS >= Meeting::consumers(comptime!(self.fillers)) {
            Role::new_Fill()
        } else {
            Role::new_Compute()
        }
    }

    /// Slot `index`, which every caller knows at comptime: the stages' schedule unrolls over its
    /// slots even when the walk itself stays rolled.
    pub fn slot_mut(&mut self, #[comptime] index: usize) -> &mut Slot<T> {
        self.slots.index_mut(index)
    }
}

/// How stages fill their slots from their own sources, at expand level: what
/// [`pipelined`](Stages::pipelined) needs of them, whatever their payload holds.
pub trait StagesFill {
    /// Whether any operand's window is fixed across the walk (filled once, above the loop).
    fn has_fixed(&self, scope: &Scope) -> bool;
    /// Fill slot `slot`'s fixed operands from `region`'s window.
    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Fill slot `slot`'s streamed operands from `region`'s window.
    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Publish slot `slot`'s last fill where no later fill will (see [`Slot::publish`]).
    fn publish(&mut self, scope: &Scope, slot: usize);
}

impl<T: CubeType> Stages<T> {
    /// Walk `walk`'s regions through these stages, `compute` consuming each region out of its
    /// slot, the stages filling from their own sources. The closure sees the slot (already this
    /// region's) and the region, and reads the slot through [`consume`](Slot::consume).
    ///
    /// Unrolling is the walk's statement ([`Walk::unrolled`]): a fragment output or a fragment
    /// read needs constant coordinates, and the kernel that allocated either says so on the walk.
    ///
    /// To fill some other way -- a third operand, a transform on the way in -- keep this schedule
    /// and bring the fill: [`pipelined_with`](Stages::pipelined_with).
    pub fn pipelined<F>(&mut self, _walk: Walk, _compute: F)
    where
        F: FnMut(&mut Slot<T>, &Region),
    {
        unexpanded!()
    }

    /// [`pipelined`](Stages::pipelined) with the fill the caller's too: `fill` writes a slot for a
    /// region, `compute` reads it back.
    ///
    /// **The schedule is the part worth sharing, not the fill.** The prologue, the lap prefetching
    /// one region ahead of the one it computes, and which consume publishes a slot no later fill
    /// will, are the protocol. What a fill *does* is the kernel's, written through [`Slot::fill`].
    ///
    /// # Panics
    ///
    /// Stages with a fixed operand (one whose window the walk leaves invariant, filled once above
    /// the loop): hoisting that fill is the stages reading their own sources, which this entry
    /// hands over, so the two cannot both be true. Stream every operand, or use
    /// [`pipelined`](Stages::pipelined).
    pub fn pipelined_with<Fill, F>(&mut self, _walk: Walk, _fill: Fill, _compute: F)
    where
        Fill: FnMut(&mut Slot<T>, &Region),
        F: FnMut(&mut Slot<T>, &Region),
    {
        unexpanded!()
    }
}

impl<T: CubeType> StagesExpand<T>
where
    StagesExpand<T>: StagesFill,
{
    pub fn __expand_pipelined_method<F>(&mut self, scope: &Scope, walk: WalkExpand, compute: F)
    where
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        // The stages' own fill, which is what makes this the convenience entry.
        schedule::run(
            scope,
            walk,
            self,
            |scope, stages, slot, region| stages.fill_streamed(scope, slot, region),
            compute,
        )
    }

    pub fn __expand_pipelined_with_method<Fill, F>(
        &mut self,
        scope: &Scope,
        walk: WalkExpand,
        mut fill: Fill,
        compute: F,
    ) where
        Fill: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        assert!(
            !self.has_fixed(scope),
            "Stages::pipelined_with: these stages hold an operand the walk leaves fixed, which \
             the schedule fills once from their own sources -- the one thing a caller's fill \
             cannot be handed. Stream every operand, or use `pipelined`."
        );
        schedule::run(
            scope,
            walk,
            self,
            move |scope, stages, slot, region| {
                let slot = stages.__expand_slot_mut_method(scope, slot);
                fill(scope, slot, region)
            },
            compute,
        )
    }
}

/// The register-staged schedule, over the two operands of a contraction.
impl<Lhs: Numeric, Rhs: Numeric> Stages<OperandPair<Lhs, Rhs>> {
    /// [`pipelined`](Stages::pipelined)'s walk with each region's fill split around the
    /// contraction before it: a unit reads its share of the next region's stage into registers,
    /// the slot is contracted, and the registers are written into the next slot. The next stage's
    /// global loads are in flight while the contraction runs, rather than issued after the
    /// barrier that frees their slot.
    ///
    /// One slot costs two cube barriers a region (the contraction's reads are done, then the
    /// write is published); two or more cost one, since the slot written is one the last barrier
    /// already freed. Only one region is ever held ahead, so stages deeper than two slots are
    /// refused: their slots past the second would hold shared memory and never a stage.
    ///
    /// Only a stage copied by every unit of the cube, straight, from a plain operand, both
    /// operands holding at most [`MOST_FETCHED_SCALARS`] between them a unit, is filled this way;
    /// anything else is refused at expansion. Stages holding an operand the walk leaves fixed
    /// take [`pipelined`](Stages::pipelined)'s schedule, at any depth: the fetch moves both
    /// operands of a slot at once, and a fixed one is filled once, above the loop.
    pub fn prefetched<F>(&mut self, _walk: Walk, _compute: F)
    where
        F: FnMut(&mut Slot<OperandPair<Lhs, Rhs>>, &Region),
    {
        unexpanded!()
    }
}

impl<Lhs: Numeric, Rhs: Numeric> StagesExpand<OperandPair<Lhs, Rhs>> {
    pub fn __expand_prefetched_method<F>(&mut self, scope: &Scope, walk: WalkExpand, compute: F)
    where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair<Lhs, Rhs>>, &RegionExpand),
    {
        // An operand the walk leaves fixed is filled once, above the loop: there is no next
        // region of it to fetch, and the ordinary schedule is the whole walk.
        if self.has_fixed(scope) {
            return self.__expand_pipelined_method(scope, walk, compute);
        }
        self.__expand_assert_copied_by_every_unit_method(scope);
        assert!(
            self.depth <= 2,
            "Stages::prefetched: {} slots hold one region ahead, so the slots past a second \
             would sit idle; build them with a depth of 1 or 2",
            self.depth
        );
        let total = walk.__expand_total_method(scope);
        self.prime(scope, &walk, &total);
        self.prefetching_laps(scope, walk, total, compute);
    }

    /// The first region, filled and published before any contraction: what the first lap reads
    /// while it fetches the second.
    fn prime(&mut self, scope: &Scope, walk: &WalkExpand, total: &NativeExpand<usize>) {
        let any = 0usize.into_expand(scope).__expand_lt_method(scope, total);
        if_expand(scope, any, |scope| {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            self.fill_streamed(scope, FIRST_SLOT, &first);
            self.publish(scope, FIRST_SLOT);
        });
    }

    /// The walk itself: each lap fetches the next region into registers, contracts the slot it
    /// already holds, then writes the registers into the slot the contraction just freed.
    fn prefetching_laps<F>(
        &mut self,
        scope: &Scope,
        walk: WalkExpand,
        total: NativeExpand<usize>,
        mut compute: F,
    ) where
        F: FnMut(&Scope, &mut SlotExpand<OperandPair<Lhs, Rhs>>, &RegionExpand),
    {
        let depth = self.depth;
        let unroll = walk.unroll;
        let (mut lhs, mut rhs) = self.__expand_fetch_buffers_method(scope);
        let laps = total
            .__expand_plus_method(scope, (depth - 1).into_expand(scope))
            .__expand_divided_by_method(scope, depth.into_expand(scope));
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let target = (j + 1) % depth;
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                let next = region_idx.__expand_plus_method(scope, 1usize.into_expand(scope));
                let in_walk = region_idx.__expand_lt_method(scope, &total);
                if_expand(scope, in_walk, |scope| {
                    let prefetching = next.__expand_lt_method(scope, &total);
                    if_expand(scope, prefetching, |scope| {
                        let upcoming = walk.__expand_region_method(scope, next);
                        self.__expand_fetch_method(scope, target, &upcoming, &mut lhs, &mut rhs);
                    });
                    let region = walk.__expand_region_method(scope, region_idx);
                    let slot = self.__expand_slot_mut_method(scope, j);
                    compute(scope, slot, &region);
                    if_expand(scope, prefetching, |scope| {
                        // One slot: every unit has read it before any overwrites it.
                        if depth == 1 {
                            self.publish(scope, FIRST_SLOT);
                        }
                        self.__expand_store_method(scope, target, &lhs, &rhs);
                        self.publish(scope, target);
                    });
                });
            }
        };
        let range = RangeExpand::new(0usize.into_expand(scope), laps);
        if unroll {
            range.expand_unroll(scope, &mut body);
        } else {
            range.expand(scope, &mut body);
        }
    }
}

/// The one schedule both entries drive: prologue, then a lap that prefetches one region ahead of
/// the one it computes, with the drain publishing what no later fill will.
///
/// `fill` writes slot `slot` for `region`; `compute` reads it back.
mod schedule {
    use super::*;

    pub(super) fn run<T: CubeType, Fill, F>(
        scope: &Scope,
        walk: WalkExpand,
        stages: &mut StagesExpand<T>,
        mut fill: Fill,
        mut compute: F,
    ) where
        StagesExpand<T>: StagesFill,
        Fill: FnMut(&Scope, &mut StagesExpand<T>, usize, &RegionExpand),
        F: FnMut(&Scope, &mut SlotExpand<T>, &RegionExpand),
    {
        let depth = stages.depth;
        let unroll = walk.unroll;
        let total = walk.__expand_total_method(scope);

        if stages.has_fixed(scope) {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            for slot in 0..depth {
                stages.fill_fixed(scope, slot, &first);
            }
        }

        // Prime every slot but the last, which the first lap's prefetch fills.
        for slot in 0..depth - 1 {
            let index = slot.into_expand(scope);
            let cond = index.__expand_lt_method(scope, &total);
            if_expand(scope, cond, |scope| {
                let region = walk.__expand_region_method(scope, slot.into_expand(scope));
                fill(scope, stages, slot, &region);
            });
        }

        let laps = total
            .__expand_plus_method(scope, (depth - 1).into_expand(scope))
            .__expand_divided_by_method(scope, depth.into_expand(scope));
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_times_method(scope, depth.into_expand(scope))
                    .__expand_plus_method(scope, j.into_expand(scope));
                if depth == 1 {
                    // Nothing is in flight: this region's fill is the last event before its read.
                    let region = walk.__expand_region_method(scope, region_idx);
                    fill(scope, stages, FIRST_SLOT, &region);
                    stages.publish(scope, FIRST_SLOT);
                    let slot = stages.__expand_slot_mut_method(scope, FIRST_SLOT);
                    compute(scope, slot, &region);
                } else {
                    let ahead =
                        region_idx.__expand_plus_method(scope, (depth - 1).into_expand(scope));
                    let prefetching = ahead.__expand_lt_method(scope, &total);
                    let in_walk = region_idx.__expand_lt_method(scope, &total);
                    // The compute is emitted once, after the branch on the fill: it is the whole
                    // unrolled contraction, its accumulator live across whatever branch holds it.
                    if_else_expand(scope, prefetching, |scope| {
                        let prefetch = walk.__expand_region_method(scope, ahead);
                        fill(scope, stages, (j + depth - 1) % depth, &prefetch);
                    })
                    .or_else(scope, |scope| {
                        // The walk is draining: no fill follows, so this consume publishes.
                        if_expand(scope, in_walk, |scope| {
                            stages.publish(scope, j);
                        });
                    });
                    if_expand(scope, in_walk, |scope| {
                        let region = walk.__expand_region_method(scope, region_idx);
                        let slot = stages.__expand_slot_mut_method(scope, j);
                        compute(scope, slot, &region);
                    });
                }
            }
        };
        let range = RangeExpand::new(0usize.into_expand(scope), laps);
        if unroll {
            range.expand_unroll(scope, &mut body);
        } else {
            range.expand(scope, &mut body);
        }
    }
}
