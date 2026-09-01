//! The contraction over an instance's share of work a level distributes as one
//! ([`LevelCuts::distribute`](crate::LevelCuts::distribute)): a share is a range of the index
//! those axes make together, so an instance contracts several output regions whole and part of
//! the two at either end.
//!
//! The share is walked as a nest, and that is the whole reason to distribute one: a register
//! accumulator opens per output region, folds that region's part of the share, and drains once.
//! Both trip counts are runtime and no drain is a runtime decision, so the accumulator keeps the
//! lexical scope every other residence has. Contracting straight into the destination instead
//! would be correct and would fold once per step rather than once per region.

use cubecl::prelude::*;

use crate::*;

/// `c = a · b` over this instance's share of the work.
///
/// The destination is written by several instances and none of them holds a whole cell, so it
/// must be one that folds ([`Write::Accumulate`]); [`Tile::accumulate`] refuses the rest before
/// this is reached.
#[cube]
pub(crate) fn distributed_mm<EA: Numeric, Out: Numeric, Lhs: Numeric, Rhs: Numeric>(
    sink: &mut Tile<Out>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    #[comptime] monoid: Monoid,
    #[comptime] semiring: Semiring,
) {
    // The regions are walked here rather than through a ring, so a residence stated at this
    // level would materialize nothing. Refused rather than ignored.
    let lhs_stage = lhs.stage_plan();
    let rhs_stage = rhs.stage_plan();
    comptime!(refuse_a_stage(lhs_stage.head(), "lhs"));
    comptime!(refuse_a_stage(rhs_stage.head(), "rhs"));

    // Nothing under this reads a second statement: the share's regions are walked here and the
    // level below runs through the ordinary schedule, which deals every axis on its own.
    comptime!(refuse_work_below(&sink.space));

    let rides = comptime!(distributed_scope(&sink.space));
    let instances = comptime!(distributed_instances(&sink.space));
    // How many steps one region of this level costs. Comptime: a divided space is fully static
    // whatever the parent was, so the stride is a constant however dynamic the index's length is.
    let stride = comptime!(steps_below(&Space::merge(&[&lhs.space, &rhs.space])));

    let regions = Walk::over(sink.op_space(lhs, rhs));
    let work = regions.total() * stride;

    // This instance's share. Two divisions rather than a size each: the shares abut, cover the
    // work once, and differ in length by at most one, with no remainder to hand out by hand.
    let pos = hardware_pos(comptime!(rides));
    let start = pos * work / instances;
    let end = (pos + 1) * work / instances;

    // An instance with no share at all, which work shorter than the device is full of.
    if start < end {
        let first = start / stride;
        // Through the region the share's last step falls in. `end` is exclusive and the share is
        // not empty, so the step before it is the one to find.
        let touched = (end - 1) / stride + 1 - first;
        let walk = regions.window(first, touched);

        for step in 0..touched {
            let region = walk.region(step);
            let base = (first + step) * stride;
            // Whole in the middle of the share, part of a region at either end of it.
            let from = select(base < start, start - base, 0);
            let to = select(end < base + stride, end - base, stride);

            let mut tile = sink.at(&region);
            let lhs = lhs.at(&region);
            let rhs = rhs.at(&region);

            let mut acc = tile.register_partition::<EA, Lhs>(&lhs, monoid);
            acc.init_identity(monoid);
            acc.mma_window(&lhs, &rhs, from, to - from, semiring);
            acc.drain_cast_into(&mut tile);
        }
    }
}

/// The scope the shares ride. Panics on a level that distributes nothing as one, which nothing
/// reaches: [`Tile::accumulate`] reads the same statement to open this scope in the first place.
fn distributed_scope(space: &Space) -> ComputeScope {
    work(space).scope()
}

/// How many instances the work is shared between.
fn distributed_instances(space: &Space) -> usize {
    work(space).instances()
}

/// An operand materialized at the level that distributes the work. Nothing stages it: the shares
/// are walked here, region by region, and the ring that would fill a slot belongs to the level
/// below. State the residence there instead.
fn refuse_a_stage(residence: Residence, operand: &str) {
    assert!(
        residence == Residence::InPlace,
        "distributed_mm: the {operand} states {residence:?} at the level that distributes the \
         work, where nothing would stage it; state it at the level below"
    );
}

/// A second distribution one level down, which the schedule under a share never reads. Two
/// scopes sharing one contraction is a real thing to want and is not this: cut an axis across the
/// inner scope instead, which the walk under a share honours like any other.
fn refuse_work_below(space: &Space) {
    let child = space.divide();
    assert!(
        child.is_final() || child.partitioner().work().is_none(),
        "distributed_mm: the level under this one distributes work too, and a share's own walk \
         deals every axis on its own. Cut an axis across that scope (`Cut::plane`, `Cut::unit`) \
         instead."
    );
}

fn work(space: &Space) -> &Work {
    space
        .partitioner()
        .work()
        .expect("distributed_mm: this level deals every axis on its own, distributing nothing")
}

/// How many steps the level under this one walks, which is what one region of this level costs
/// the share.
///
/// One instance's steps, not the grid's: an axis that level cuts across planes or lanes is
/// covered by the instances of a *cube*, which all step together, so the share is counted in the
/// steps they take together rather than in the tiles they cover between them.
fn steps_below(space: &Space) -> usize {
    let child = space.divide();
    child
        .axes()
        .map(|axis| {
            let grid = child.count(axis);
            match child.partitioner().distribution(axis) {
                Distribution::Sequential => grid,
                Distribution::Spatial { coverage, .. } => grid / coverage.instances(grid),
            }
        })
        .product()
}

#[cfg(test)]
mod tests {
    use super::refuse_work_below;
    use crate::{Axis, Buffering, CubeAxis, Cut, Space, Tiling, WalkOrder, cubes};

    const M: Axis = Axis(0);
    const N: Axis = Axis(1);
    const K: Axis = Axis(2);

    // Host-side, because a comptime panic raised in a kernel lands on a worker thread where
    // `#[should_panic]` never sees it and the launch returns zeros.

    fn space(twice: bool) -> Space {
        Tiling::new()
            .extents(&[(M, 8), (N, 8), (K, 8)])
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| {
                l.distribute(&[(M, 4), (N, 4), (K, 8)], cubes(CubeAxis::X).instances(3))
            })
            .level(WalkOrder::RowMajor, Buffering::SINGLE, |l| match twice {
                true => l.distribute(&[(M, 4), (N, 4), (K, 4)], cubes(CubeAxis::Y).instances(2)),
                false => l
                    .axis(M, Cut::sequential(4))
                    .axis(N, Cut::sequential(4))
                    .axis(K, Cut::sequential(4)),
            })
            .build()
    }

    #[test]
    #[should_panic = "distributes work too"]
    fn a_second_distribution_under_a_share_is_refused() {
        refuse_work_below(&space(true));
    }

    #[test]
    fn an_ordinary_level_under_a_share_is_the_walk_it_always_was() {
        refuse_work_below(&space(false));
    }
}
