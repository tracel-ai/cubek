//! The one buffered walk: `depth` [`Staging`] slots driven as a circular software pipeline, so
//! `depth - 1` regions are in flight while one computes.
//!
//! Single and double buffering are this schedule at `depth` 1 and 2, not two hand-written walks.
//! At depth 1 a region's fill is the last event before its own read, so the consume publishes it;
//! at any greater depth the fill that runs one lap ahead does. That difference, the prologue, and
//! the drain are the whole protocol, and it lives here once. An operation supplies only what a
//! slot holds and what consuming one computes, through [`Pipelined`].

use cubecl::prelude::*;

use crate::*;

/// The `depth` slots of one buffered walk.
#[derive(CubeType)]
pub struct Ring<T: CubeType> {
    pub(crate) slots: Sequence<Staging<T>>,
    #[cube(comptime)]
    pub(crate) depth: usize,
}

#[cube]
impl<T: CubeType> Ring<T> {
    /// Wrap already-built slots. The public entries are the operand-deducing
    /// [`binary`](Ring::binary) / [`unary`](Ring::unary).
    pub(crate) fn wrap(slots: Sequence<Staging<T>>, #[comptime] depth: usize) -> Ring<T> {
        comptime!(assert!(
            depth > 0,
            "Ring: a pipeline needs at least one slot"
        ));
        Ring::<T> { slots, depth }
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

    /// Whether any slot reads an operand by selecting fragments, which takes comptime coordinates
    /// and so stands up only under an unrolled walk. Uniform across slots since all slots share one plan.
    pub(crate) fn has_fragment_read(&self) -> comptime_type!(bool) {
        self.slots.index(FIRST_SLOT).has_fragment_read()
    }
}

/// The operation-specific half of a buffered walk. Everything about *when* a slot is filled,
/// consumed, or published is [`pipelined_walk`]'s; an implementor says only what a slot holds and
/// what filling or consuming one means.
#[cube]
pub trait Pipelined: CubeType {
    /// The operand bundle one slot stages.
    type Slot: CubeType;

    /// Build the walk's `depth` slots over this operation's operands.
    fn ring(
        &self,
        #[comptime] op_space: Space,
        #[comptime] out: Space,
        #[comptime] depth: usize,
    ) -> Ring<Self::Slot>;

    /// Whether the walk must unroll, so every region's coordinates fold to comptime constants.
    /// Asked once, of the ring the operation just built.
    fn unrolled(&self, ring: &Ring<Self::Slot>) -> comptime_type!(bool);

    /// Fill the operands whose window the walk leaves fixed. Runs once per slot, above the loop.
    fn fill_fixed(&self, slot: &mut Staging<Self::Slot>, region: &Region);

    /// Fill the operands the walk moves, for `region`.
    fn fill_streamed(&self, slot: &mut Staging<Self::Slot>, region: &Region);

    /// Consume `region` out of `slot`. `publish` when no later fill will publish this one, so the
    /// consume has to.
    fn compute(
        &mut self,
        slot: &mut Staging<Self::Slot>,
        region: &Region,
        #[comptime] publish: bool,
    );
}

/// Walk `op_space` region by region through a ring of `depth` slots.
///
/// The prologue primes slots `0..depth - 1` with the first `depth - 1` regions. Lap `p` then
/// visits regions `p * depth + j` for each slot `j`: it fills the region one lap ahead into the
/// slot that just came free, then consumes this one. Both slot indices are comptime (`j` and
/// `(j + depth - 1) % depth`), which is what lets one rolled loop drive a ring of any depth.
///
/// A consume publishes its own fill when no fill follows it: always at `depth == 1`, where the
/// fill and the read of a region are adjacent, and at the walk's last `depth - 1` regions, where
/// there is nothing left to prefetch.
#[cube]
pub(crate) fn pipelined_walk<P: Pipelined>(
    op: &mut P,
    op_space: Space,
    #[comptime] out: Space,
    #[comptime] depth: usize,
) {
    let mut ring = op.ring(comptime!(op_space.clone()), out, depth);
    // Re-bound through `comptime!`: `#[unroll(flag)]` only unrolls when the macro can see `flag`
    // as a comptime binding, and silently rolls the loop otherwise.
    let unrolled = op.unrolled(&ring);
    let unroll = comptime!(unrolled);
    let has_fixed = ring.has_fixed();
    let walk = Walk::over(op_space);
    let total = walk.total();

    // A fixed operand's window never moves, so region 0's is every region's. Later slots reusing
    // the first slot's buffer for it read `WindowMode::Reused` and skip the copy.
    if comptime!(has_fixed) {
        let first = walk.region(FIRST_SLOT);
        #[unroll]
        for slot in 0..depth {
            op.fill_fixed(ring.slot_mut(slot), &first);
        }
    }

    // Prime every slot but the last, which the first lap's prefetch fills.
    #[unroll]
    for slot in 0..comptime!(depth - 1) {
        if slot < total {
            let region = walk.region(slot);
            op.fill_streamed(ring.slot_mut(slot), &region);
        }
    }

    let laps = total.fadd(comptime!(depth - 1)).fdiv(depth);
    #[unroll(unroll)]
    for lap in 0..laps {
        #[unroll]
        for j in 0..depth {
            let region_idx = lap.fmul(depth).fadd(j);
            if comptime!(depth == 1) {
                // Nothing is in flight: this region's fill is the last event before its read.
                let region = walk.region(region_idx);
                op.fill_streamed(ring.slot_mut(FIRST_SLOT), &region);
                op.compute(ring.slot_mut(FIRST_SLOT), &region, true);
            } else {
                let ahead = region_idx.fadd(comptime!(depth - 1));
                if ahead < total {
                    let prefetch = walk.region(ahead);
                    op.fill_streamed(ring.slot_mut(comptime!((j + depth - 1) % depth)), &prefetch);
                    let region = walk.region(region_idx);
                    op.compute(ring.slot_mut(j), &region, false);
                } else if region_idx < total {
                    // The walk is draining: no fill follows, so this consume publishes.
                    let region = walk.region(region_idx);
                    op.compute(ring.slot_mut(j), &region, true);
                }
            }
        }
    }
}

/// Whether a buffered walk must unroll, the one answer every [`Pipelined`] operation gives the
/// same way: the level *cuts* a fragment-partition output (each region selects its block by
/// comptime coordinate), or a slot reads an operand as fragments, selected the same way and
/// standing up only when `op_space` is itself static-walkable. An smem-staged level stays rolled:
/// unrolling would re-stage its shared memory per copy.
///
/// The output and the operands are asked separately because either can be fragments on its own:
/// a level can cut a fragment output over memory operands, or read a resident fragment into a
/// memory output.
///
/// Operations differ only in which space is the operand merge, so they pass it in.
#[cube]
pub(crate) fn stage_walk_unrolled<Acc: Numeric>(
    acc: &Tile<Acc>,
    #[comptime] op_space: Space,
    #[comptime] has_fragment_read: bool,
) -> comptime_type!(bool) {
    let cuts = acc.tile_kind.cuts_partition(comptime!(acc.space.clone()));
    comptime!({
        let unroll = cuts || (has_fragment_read && op_space.static_walkable());
        if unroll {
            assert!(
                op_space.is_static(),
                "stage_walk_unrolled: an unrolled walk must be static (dynamic extents cannot fold to comptime coordinates)"
            );
        }
        unroll
    })
}
