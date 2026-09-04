//! The one buffered walk: `depth` [`Staging`] slots driven as a circular software pipeline, so
//! `depth - 1` regions are in flight while one computes.
//!
//! Single and double buffering are this schedule at `depth` 1 and 2, not two hand-written walks.
//! At depth 1 a region's fill is the last event before its own read, so the consume publishes it;
//! at any greater depth the fill that runs one lap ahead does. That difference, the prologue, and
//! the drain are the whole protocol, and it lives here once ([`pipelined`]). A kernel supplies
//! only the ring and what consuming a slot computes.

use cubecl::frontend::branch::{if_else_expand, if_expand};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// The `depth` slots of one buffered walk, and the operands they are filled from: the same
/// payload shape at this level, so [`pipelined`] can fill a slot for a region on its own.
#[derive(CubeType)]
pub struct Ring<T: CubeType> {
    pub(crate) slots: Sequence<Staging<T>>,
    pub(crate) sources: T,
    #[cube(comptime)]
    pub(crate) depth: usize,
}

#[cube]
impl<T: CubeType> Ring<T> {
    /// Wrap already-built slots over the operands they stage. The public entries are
    /// [`smem`](Ring::smem) and [`smem_single`](Ring::smem_single).
    pub(crate) fn wrap(
        slots: Sequence<Staging<T>>,
        sources: T,
        #[comptime] depth: usize,
    ) -> Ring<T> {
        comptime!(assert!(
            depth > 0,
            "Ring: a pipeline needs at least one slot"
        ));
        Ring::<T> {
            slots,
            sources,
            depth,
        }
    }

    /// Slot `index`, which every caller knows at comptime: the ring's schedule unrolls over its
    /// slots even when the walk itself stays rolled.
    pub fn slot_mut(&mut self, #[comptime] index: usize) -> &mut Staging<T> {
        self.slots.index_mut(index)
    }

    /// Whether any operand across the ring is fixed across the walk. Uniform across slots since
    /// all slots share one plan.
    pub(crate) fn has_fixed(&self) -> comptime_type!(bool) {
        self.slots.index(FIRST_SLOT).has_fixed()
    }
}

/// How a ring fills its slots from its own sources, at expand level: what [`pipelined`] needs
/// from a payload shape, implemented once per shape beside the ring constructors
/// ([`fill`](crate::fill)).
pub trait RingFill {
    /// Whether any operand's window is fixed across the walk (filled once, above the loop).
    fn has_fixed(&self, scope: &Scope) -> bool;
    /// Fill slot `slot`'s fixed operands from `region`'s window.
    fn fill_fixed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Fill slot `slot`'s streamed operands from `region`'s window.
    fn fill_streamed(&mut self, scope: &Scope, slot: usize, region: &RegionExpand);
    /// Publish slot `slot`'s last fill where no later fill will (see [`Staging::publish`]).
    fn publish(&mut self, scope: &Scope, slot: usize);
}

/// Walk `walk`'s regions through `ring`, `compute` consuming each region out of its slot. The
/// ring knows its sources, so every fill is the ring's; the closure sees the slot (already this
/// region's) and the region, and reads the slot through [`consume`](Staging::consume).
///
/// Unrolling is the walk's statement ([`Walk::unrolled`]): a fragment output or a fragment read
/// needs constant coordinates, and the kernel that allocated either says so on the walk.
pub fn pipelined<T: CubeType, F>(_walk: Walk, _ring: &mut Ring<T>, _compute: F)
where
    F: FnMut(&mut Staging<T>, &Region),
{
    unexpanded!()
}

/// The expand of [`pipelined`], spelled at expand level so the compute body can be a closure.
pub mod pipelined {
    use super::*;

    pub fn expand<T: CubeType, F>(
        scope: &Scope,
        walk: WalkExpand,
        ring: &mut RingExpand<T>,
        mut compute: F,
    ) where
        RingExpand<T>: RingFill,
        F: FnMut(&Scope, &mut StagingExpand<T>, &RegionExpand),
    {
        let depth = ring.depth;
        let unroll = walk.unroll;
        let total = walk.__expand_total_method(scope);

        if ring.has_fixed(scope) {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            for slot in 0..depth {
                ring.fill_fixed(scope, slot, &first);
            }
        }

        // Prime every slot but the last, which the first lap's prefetch fills.
        for slot in 0..depth - 1 {
            let index = slot.into_expand(scope);
            let cond = index.__expand_lt_method(scope, &total);
            if_expand(scope, cond, |scope| {
                let region = walk.__expand_region_method(scope, slot.into_expand(scope));
                ring.fill_streamed(scope, slot, &region);
            });
        }

        let laps = total
            .__expand_fadd_method(scope, (depth - 1).into_expand(scope))
            .__expand_fdiv_method(scope, depth.into_expand(scope));
        let body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_fmul_method(scope, depth.into_expand(scope))
                    .__expand_fadd_method(scope, j.into_expand(scope));
                if depth == 1 {
                    // Nothing is in flight: this region's fill is the last event before its read.
                    let region = walk.__expand_region_method(scope, region_idx);
                    ring.fill_streamed(scope, FIRST_SLOT, &region);
                    ring.publish(scope, FIRST_SLOT);
                    let slot = ring.__expand_slot_mut_method(scope, FIRST_SLOT);
                    compute(scope, slot, &region);
                } else {
                    let ahead =
                        region_idx.__expand_fadd_method(scope, (depth - 1).into_expand(scope));
                    let prefetching = ahead.__expand_lt_method(scope, &total);
                    let draining = region_idx.__expand_lt_method(scope, &total);
                    if_else_expand(scope, prefetching, |scope| {
                        let prefetch = walk.__expand_region_method(scope, ahead);
                        ring.fill_streamed(scope, (j + depth - 1) % depth, &prefetch);
                        let region = walk.__expand_region_method(scope, region_idx);
                        let slot = ring.__expand_slot_mut_method(scope, j);
                        compute(scope, slot, &region);
                    })
                    .or_else(scope, |scope| {
                        // The walk is draining: no fill follows, so this consume publishes.
                        if_expand(scope, draining, |scope| {
                            let region = walk.__expand_region_method(scope, region_idx);
                            ring.publish(scope, j);
                            let slot = ring.__expand_slot_mut_method(scope, j);
                            compute(scope, slot, &region);
                        });
                    });
                }
            }
        };
        let range = RangeExpand::new(0usize.into_expand(scope), laps);
        if unroll {
            range.expand_unroll(scope, body);
        } else {
            range.expand(scope, body);
        }
    }
}
