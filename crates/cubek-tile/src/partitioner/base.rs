//! The [`Partitioner`]: a descent strategy for one level of the space.

use cubecl::prelude::*;

use crate::{Axis, ByAxis, Grid};

use super::{Distribution, Walk, WalkOrder};

/// A descent strategy for one level of the space: the split (per-axis sub-tile
/// size + [`Distribution`]) and the [`WalkOrder`]. Comptime fields answered as
/// methods.
#[derive(CubeType, CubeLaunch, Clone, PartialEq, Eq, Hash, Debug)]
#[expand(derive(Clone))]
pub struct Partitioner {
    #[cube(comptime)]
    sub_tile: ByAxis<usize>,
    #[cube(comptime)]
    dists: ByAxis<Distribution>,
    #[cube(comptime)]
    order: WalkOrder,
}

impl Partitioner {
    /// A partitioner with the given per-axis split and walk [`order`](WalkOrder).
    /// Concrete orders are constructed via the conveniences in
    /// [`walk_order`](super::walk_order) (`row_major`, `reversed`).
    pub fn new(sub_tile: ByAxis<usize>, dists: ByAxis<Distribution>, order: WalkOrder) -> Self {
        Partitioner {
            sub_tile,
            dists,
            order,
        }
    }

    /// The launch arg for carrying this partitioner on a tile.
    pub fn launch<R: Runtime>(&self) -> PartitionerLaunch<R> {
        PartitionerLaunch::new(self.sub_tile.clone(), self.dists.clone(), self.order)
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

    /// The [`Walk`] over `grid`. The caller supplies the grid; the partitioner
    /// owns only the order it's walked in.
    pub fn walk(&self, grid: Grid) -> Walk {
        Walk::new(grid, self.clone())
    }
}
