//! The one buffered walk: `depth` [`Staging`] slots driven as a circular software pipeline, so
//! `depth - 1` regions are in flight while one computes.
//!
//! Single and double buffering are this schedule at `depth` 1 and 2. At depth 1 a region's fill is
//! the last event before its read, so the consume publishes it; deeper, the fill a lap ahead does.
//! That, prologue and drain are the protocol ([`pipelined`]); a kernel supplies ring and consume.
//!
//! [`pipelined_through_registers`] is the same walk with each region's fill split around the
//! contraction before it, the next region's loads held in the units' registers meanwhile.

use cubecl::frontend::branch::{if_else_expand, if_expand};
use cubecl::ir::Scope;
use cubecl::prelude::*;
use cubecl::unexpanded;

use crate::*;

/// What a plane of the cube does with a ring's slots ([`Ring::role`]).
#[derive(CubeType, CubeTypeMut, IntoRuntime)]
#[cube(runtime_variants)]
pub enum Role {
    /// Fills the slots, and takes no tile of any level ([`Level::filled_by`]).
    Fill,
    /// Reads the slots and computes out of them, and fills none.
    Compute,
}

/// The `depth` slots of one buffered walk, and the operands they are filled from: the same
/// payload shape at this level, so [`pipelined`] can fill a slot for a region on its own.
#[derive(CubeType)]
pub struct Ring<T: CubeType> {
    pub(crate) slots: Sequence<Staging<T>>,
    pub(crate) sources: T,
    // Read by the schedules ([`pipelined`], [`pipelined_through_registers`]) at expand level only.
    #[allow(dead_code)]
    #[cube(comptime)]
    pub(crate) depth: usize,
    /// Planes of the cube that fill these slots and do nothing else ([`Level::filled_by`]).
    #[cube(comptime)]
    pub(crate) fillers: usize,
}

#[cube]
impl<T: CubeType> Ring<T> {
    /// Wrap already-built slots over the operands they stage. The public entries are
    /// [`smem`](Ring::smem) and [`smem_single`](Ring::smem_single).
    pub(crate) fn wrap(
        slots: Sequence<Staging<T>>,
        sources: T,
        #[comptime] depth: usize,
        #[comptime] fillers: usize,
    ) -> Ring<T> {
        comptime!(assert!(
            depth > 0,
            "Ring: a pipeline needs at least one slot"
        ));
        Ring::<T> {
            slots,
            sources,
            depth,
            fillers,
        }
    }

    /// What this plane does with the ring's slots. The planes a walk sets aside to fill sit at
    /// the end of the cube ([`Level::filled_by`]), so a unit fills exactly when it stands at or
    /// past the ones that compute, and every plane below is the one it would have been.
    ///
    /// Where the walk set none aside, every plane computes and fills its own slots, which is the
    /// schedule [`pipelined`] writes.
    pub fn role(&self) -> Role {
        if comptime!(self.fillers == 0) {
            Role::new_Compute()
        } else if UNIT_POS >= Pipeline::consumers(comptime!(self.fillers)) {
            Role::new_Fill()
        } else {
            Role::new_Compute()
        }
    }

    /// Slot `index`, which every caller knows at comptime: the ring's schedule unrolls over its
    /// slots even when the walk itself stays rolled.
    pub fn slot_mut(&mut self, #[comptime] index: usize) -> &mut Staging<T> {
        self.slots.index_mut(index)
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

/// Walk `walk`'s regions through `ring`, `compute` consuming each region out of its slot, the
/// ring filling from its own sources. The closure sees the slot (already this region's) and the
/// region, and reads the slot through [`consume`](Staging::consume).
///
/// Unrolling is the walk's statement ([`Walk::unrolled`]): a fragment output or a fragment read
/// needs constant coordinates, and the kernel that allocated either says so on the walk.
///
/// To fill some other way — a third operand, a transform on the way in — keep this schedule and
/// bring the fill: [`pipelined_with`].
pub fn pipelined<T: CubeType, F>(_walk: Walk, _ring: &mut Ring<T>, _compute: F)
where
    F: FnMut(&mut Staging<T>, &Region),
{
    unexpanded!()
}

/// [`pipelined`] with the fill the caller's too: `fill` writes a slot for a region, `compute`
/// reads it back.
///
/// **The schedule is the part worth sharing, not the fill.** The prologue, the lap prefetching
/// one region ahead of the one it computes, and which consume publishes a slot no later fill
/// will, are the protocol. What a fill *does* is the kernel's, written through [`Staging::fill`].
///
/// # Panics
///
/// A ring with a fixed operand (one whose window the walk leaves invariant, filled once above
/// the loop): hoisting that fill is the ring reading its own sources, which this entry hands
/// over, so the two cannot both be true. Stream every operand, or use [`pipelined`].
pub fn pipelined_with<T: CubeType, Fill, F>(
    _walk: Walk,
    _ring: &mut Ring<T>,
    _fill: Fill,
    _compute: F,
) where
    Fill: FnMut(&mut Staging<T>, &Region),
    F: FnMut(&mut Staging<T>, &Region),
{
    unexpanded!()
}

/// [`pipelined`]'s walk with each region's fill split around the contraction before it: a unit
/// reads its share of the next region's stage into registers, the slot is contracted, and the
/// registers are written into the next slot. The next stage's global loads are in flight while
/// the contraction runs, rather than issued after the barrier that frees their slot.
///
/// One slot costs two cube barriers a region (the contraction's reads are done, then the write
/// is published); two or more cost one, since the slot written is one the last barrier already
/// freed. Only one region is ever held ahead, so slots past a second sit idle.
///
/// Only a stage copied by every unit of the cube, straight, from a plain operand, and holding at
/// most [`MOST_FETCHED_SCALARS`] of it a unit, is filled this way; anything else is refused at
/// expansion. A ring holding an operand the walk leaves fixed takes [`pipelined`]'s schedule: the
/// fetch moves both operands of a slot at once, and a fixed one is filled once, above the loop.
pub fn pipelined_through_registers<Lhs: Numeric, Rhs: Numeric, F>(
    _walk: Walk,
    _ring: &mut Ring<(Tile<Lhs>, Tile<Rhs>)>,
    _compute: F,
) where
    F: FnMut(&mut Staging<(Tile<Lhs>, Tile<Rhs>)>, &Region),
{
    unexpanded!()
}

/// The expand of [`pipelined_through_registers`].
pub mod pipelined_through_registers {
    use super::*;

    pub fn expand<Lhs: Numeric, Rhs: Numeric, F>(
        scope: &Scope,
        walk: WalkExpand,
        ring: &mut RingExpand<(Tile<Lhs>, Tile<Rhs>)>,
        mut compute: F,
    ) where
        F: FnMut(&Scope, &mut StagingExpand<(Tile<Lhs>, Tile<Rhs>)>, &RegionExpand),
    {
        ring.__expand_assert_copied_by_every_unit_method(scope);
        // An operand the walk leaves fixed is filled once, above the loop: there is no next
        // region of it to fetch, and the ring's own schedule is the whole walk.
        if ring.has_fixed(scope) {
            return super::pipelined::expand(scope, walk, ring, compute);
        }
        let depth = ring.depth;
        let unroll = walk.unroll;
        let total = walk.__expand_total_method(scope);

        // The first region, filled and published before any contraction.
        let any = 0usize.into_expand(scope).__expand_lt_method(scope, &total);
        if_expand(scope, any, |scope| {
            let first = walk.__expand_region_method(scope, FIRST_SLOT.into_expand(scope));
            ring.fill_streamed(scope, FIRST_SLOT, &first);
            ring.publish(scope, FIRST_SLOT);
        });

        let mut lhs = ring.__expand_lhs_fetch_buffer_method(scope);
        let mut rhs = ring.__expand_rhs_fetch_buffer_method(scope);
        let laps = total
            .__expand_fadd_method(scope, (depth - 1).into_expand(scope))
            .__expand_fdiv_method(scope, depth.into_expand(scope));
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let target = (j + 1) % depth;
                let region_idx = lap
                    .__expand_fmul_method(scope, depth.into_expand(scope))
                    .__expand_fadd_method(scope, j.into_expand(scope));
                let next = region_idx.__expand_fadd_method(scope, 1usize.into_expand(scope));
                let in_walk = region_idx.__expand_lt_method(scope, &total);
                if_expand(scope, in_walk, |scope| {
                    let prefetching = next.__expand_lt_method(scope, &total);
                    if_expand(scope, prefetching.clone(), |scope| {
                        let upcoming = walk.__expand_region_method(scope, next.clone());
                        ring.__expand_fetch_method(scope, target, &upcoming, &mut lhs, &mut rhs);
                    });
                    let region = walk.__expand_region_method(scope, region_idx);
                    let slot = ring.__expand_slot_mut_method(scope, j);
                    compute(scope, slot, &region);
                    if_expand(scope, prefetching, |scope| {
                        // One slot: every unit has read it before any overwrites it.
                        if depth == 1 {
                            ring.publish(scope, FIRST_SLOT);
                        }
                        ring.__expand_store_method(scope, target, &lhs, &rhs);
                        ring.publish(scope, target);
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

/// The expand of [`pipelined`], spelled at expand level so the compute body can be a closure.
pub mod pipelined {
    use super::schedule::run;
    use super::*;

    pub fn expand<T: CubeType, F>(
        scope: &Scope,
        walk: WalkExpand,
        ring: &mut RingExpand<T>,
        compute: F,
    ) where
        RingExpand<T>: RingFill,
        F: FnMut(&Scope, &mut StagingExpand<T>, &RegionExpand),
    {
        // The ring's own fill, which is what makes this the convenience entry.
        run(
            scope,
            walk,
            ring,
            |scope, ring, slot, region| ring.fill_streamed(scope, slot, region),
            compute,
        )
    }
}

/// The expand of [`pipelined_with`].
pub mod pipelined_with {
    use super::schedule::run;
    use super::*;

    pub fn expand<T: CubeType, Fill, F>(
        scope: &Scope,
        walk: WalkExpand,
        ring: &mut RingExpand<T>,
        mut fill: Fill,
        compute: F,
    ) where
        RingExpand<T>: RingFill,
        Fill: FnMut(&Scope, &mut StagingExpand<T>, &RegionExpand),
        F: FnMut(&Scope, &mut StagingExpand<T>, &RegionExpand),
    {
        assert!(
            !ring.has_fixed(scope),
            "pipelined_with: this ring holds an operand the walk leaves fixed, which the \
             schedule fills once from the ring's own sources — the one thing a caller's fill \
             cannot be handed. Stream every operand, or use `pipelined`."
        );
        run(
            scope,
            walk,
            ring,
            move |scope, ring, slot, region| {
                let slot = ring.__expand_slot_mut_method(scope, slot);
                fill(scope, slot, region)
            },
            compute,
        )
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
        ring: &mut RingExpand<T>,
        mut fill: Fill,
        mut compute: F,
    ) where
        RingExpand<T>: RingFill,
        Fill: FnMut(&Scope, &mut RingExpand<T>, usize, &RegionExpand),
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
                fill(scope, ring, slot, &region);
            });
        }

        let laps = total
            .__expand_fadd_method(scope, (depth - 1).into_expand(scope))
            .__expand_fdiv_method(scope, depth.into_expand(scope));
        let mut body = |scope: &Scope, lap: NativeExpand<usize>| {
            for j in 0..depth {
                let region_idx = lap
                    .__expand_fmul_method(scope, depth.into_expand(scope))
                    .__expand_fadd_method(scope, j.into_expand(scope));
                if depth == 1 {
                    // Nothing is in flight: this region's fill is the last event before its read.
                    let region = walk.__expand_region_method(scope, region_idx);
                    fill(scope, ring, FIRST_SLOT, &region);
                    ring.publish(scope, FIRST_SLOT);
                    let slot = ring.__expand_slot_mut_method(scope, FIRST_SLOT);
                    compute(scope, slot, &region);
                } else {
                    let ahead =
                        region_idx.__expand_fadd_method(scope, (depth - 1).into_expand(scope));
                    let prefetching = ahead.__expand_lt_method(scope, &total);
                    let in_walk = region_idx.__expand_lt_method(scope, &total);
                    // The compute is emitted once, after the branch on the fill: it is the whole
                    // unrolled contraction, its accumulator live across whatever branch holds it.
                    if_else_expand(scope, prefetching, |scope| {
                        let prefetch = walk.__expand_region_method(scope, ahead);
                        fill(scope, ring, (j + depth - 1) % depth, &prefetch);
                    })
                    .or_else(scope, |scope| {
                        // The walk is draining: no fill follows, so this consume publishes.
                        if_expand(scope, in_walk.clone(), |scope| {
                            ring.publish(scope, j);
                        });
                    });
                    if_expand(scope, in_walk, |scope| {
                        let region = walk.__expand_region_method(scope, region_idx);
                        let slot = ring.__expand_slot_mut_method(scope, j);
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
