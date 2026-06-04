//! The [`Partitioner`]: a descent strategy for one level of the space.

use cubecl::prelude::*;

use crate::{Axis, ByAxis};

use super::{Distribution, WalkOrder};

/// How a level executes its split — the move `c.mma(a, b)` picks for a whole
/// tile, a sibling of [`Distribution`] (how it spreads) and [`WalkOrder`] (in what
/// sequence). Staging-vs-not and pipeline depth ride here as one knob.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Schedule {
    /// No staging, each operand sub-tile is used straight 
    Direct,
    /// Stage each operand sub-tile
    Staged,
    /// Staged with two buffers, prefetching the next sub-tile while computing the
    /// current one.
    DoubleBuffered,
}

/// A descent strategy for one level of the space: the split (per-axis sub-tile
/// size + [`Distribution`]), the [`WalkOrder`], and the [`Schedule`]. Comptime
/// fields answered as methods.
#[derive(CubeType, CubeLaunch, Clone, PartialEq, Eq, Hash, Debug)]
#[expand(derive(Clone))]
pub struct Partitioner {
    #[cube(comptime)]
    sub_tile: ByAxis<usize>,
    #[cube(comptime)]
    dists: ByAxis<Distribution>,
    #[cube(comptime)]
    order: WalkOrder,
    #[cube(comptime)]
    schedule: Schedule,
}

impl Partitioner {
    /// A partitioner with the given per-axis split and walk [`order`](WalkOrder),
    /// [`Staged`](Schedule::Staged) by default. Concrete orders are constructed via
    /// the conveniences in [`walk_order`](super::walk_order) (`row_major`,
    /// `reversed`); override the schedule with [`direct`](Partitioner::direct) /
    /// [`double_buffered`](Partitioner::double_buffered).
    pub fn new(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>, order: WalkOrder) -> Self {
        Partitioner {
            sub_tile,
            dists,
            order,
            schedule: Schedule::Staged,
        }
    }

    /// Switch to the [`Direct`](Schedule::Direct) schedule (no staging).
    pub fn direct(mut self) -> Self {
        self.schedule = Schedule::Direct;
        self
    }

    /// Switch to the [`DoubleBuffered`](Schedule::DoubleBuffered) schedule.
    pub fn double_buffered(mut self) -> Self {
        self.schedule = Schedule::DoubleBuffered;
        self
    }

    /// The launch arg for carrying this partitioner on a tile.
    pub fn launch<R: Runtime>(&self) -> PartitionerLaunch<R> {
        PartitionerLaunch::new(
            self.sub_tile.clone(),
            self.dists.clone(),
            self.order,
            self.schedule,
        )
    }

    /// Sub-tile edge along `axis` (host) — the [`Space`]-level twin of the
    /// comptime [`sub_tile_edge`](Partitioner::sub_tile_edge). [`Space::divide`]
    /// shrinks each axis to this edge to build the child tiling level.
    pub fn edge(&self, axis: Axis) -> usize {
        self.sub_tile.get(axis)
    }
}

#[cube]
impl Partitioner {
    /// Sub-tile edge along an axis — comptime, since it sizes shared memory.
    pub fn sub_tile_edge(&self, #[comptime] axis: Axis) -> comptime_type!(usize) {
        comptime!(self.sub_tile.get(axis))
    }

    /// How an axis is distributed.
    pub fn distribution(&self, #[comptime] axis: Axis) -> comptime_type!(Distribution) {
        comptime!(self.dists.get(axis))
    }

    /// The order this partitioner visits its steps in.
    pub fn order(&self) -> comptime_type!(WalkOrder) {
        comptime!(self.order)
    }

    /// The schedule (move) a whole tile's `mma` runs.
    pub fn schedule(&self) -> comptime_type!(Schedule) {
        comptime!(self.schedule)
    }
}
