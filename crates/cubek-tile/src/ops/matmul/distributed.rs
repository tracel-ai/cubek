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

fn work(space: &Space) -> &Work {
    space
        .partitioner()
        .work()
        .expect("distributed_mm: this level deals every axis on its own, distributing nothing")
}

/// How many steps the level under this one walks, which is what one region of this level costs
/// the share.
fn steps_below(space: &Space) -> usize {
    let child = space.divide();
    child.axes().map(|axis| child.count(axis)).product()
}
