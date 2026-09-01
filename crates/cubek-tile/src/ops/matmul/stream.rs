//! The contraction a [`Streamed`](Deal::Streamed) level runs: an instance holds a run of the
//! line rather than a region of the grid, so it contracts several output tiles and only part of
//! the contraction of the two at either end.
//!
//! The run is a nest, and that is the whole reason to deal one: a register accumulator opens per
//! output tile, folds every step of that tile's share of the line, and drains once. Both trip
//! counts are runtime and neither drain is a runtime decision, so the accumulator keeps the
//! lexical scope every other residence has. Contracting straight into the destination instead
//! would be correct and would fold once per step rather than once per tile.

use cubecl::prelude::*;

use crate::*;

/// `c = a · b` where this instance holds a run of the line.
///
/// The destination is written by several instances and none of them holds a whole cell, so it
/// must be one that folds ([`Write::Accumulate`]); [`Tile::accumulate`] refuses the rest before
/// this is reached.
#[cube]
pub(crate) fn stream_mm<EA: Numeric, Out: Numeric, Lhs: Numeric, Rhs: Numeric>(
    sink: &mut Tile<Out>,
    lhs: &Tile<Lhs>,
    rhs: &Tile<Rhs>,
    #[comptime] monoid: Monoid,
    #[comptime] semiring: Semiring,
) {
    let rides = comptime!(streamed_scope(&sink.space));
    let instances = comptime!(streamed_instances(&sink.space));
    // How many steps one region of this level costs. Comptime: a divided space is fully static
    // whatever the parent was, so the line's stride is a constant however dynamic its length is.
    let stride = comptime!(steps_below(&Space::merge(&[&lhs.space, &rhs.space])));

    let regions = Walk::over(sink.op_space(lhs, rhs));
    let line = regions.total() * stride;

    // This instance's run. Two divisions rather than a share each: the runs abut, cover the line
    // once, and differ in length by at most one, with no remainder to hand out by hand.
    let pos = hardware_pos(comptime!(rides));
    let start = pos * line / instances;
    let end = (pos + 1) * line / instances;

    // An instance with no run at all, which a line shorter than the device is full of.
    if start < end {
        let first = start / stride;
        // Through the region the run's last step falls in. `end` is exclusive and the run is not
        // empty, so the step before it is the one to find.
        let touched = (end - 1) / stride + 1 - first;
        let walk = regions.window(first, touched);

        for step in 0..touched {
            let region = walk.region(step);
            let base = (first + step) * stride;
            // Whole in the middle of the run, part of a region at either end of it.
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

/// The scope a streamed level's runs ride. Panics on a level dealt per axis, which nothing
/// reaches: [`Tile::accumulate`] reads the same deal to pick this scope in the first place.
fn streamed_scope(space: &Space) -> ComputeScope {
    match space.partitioner().deal() {
        Deal::Streamed { scope, .. } => scope,
        Deal::PerAxis => panic!("stream_mm: this level deals its grid per axis, not as a line"),
    }
}

/// How many runs the line is cut into.
fn streamed_instances(space: &Space) -> usize {
    match space.partitioner().deal() {
        Deal::Streamed { instances, .. } => instances,
        Deal::PerAxis => panic!("stream_mm: this level deals its grid per axis, not as a line"),
    }
}

/// How many steps the level under this one walks, which is what one region of this level costs
/// the run.
fn steps_below(space: &Space) -> usize {
    let child = space.divide();
    child.axes().map(|axis| child.count(axis)).product()
}
