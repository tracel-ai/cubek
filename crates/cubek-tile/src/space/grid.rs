//! The runtime extent of a [`Space`], measured in tiles.

use cubecl::prelude::*;

use super::{Axis, Space};

/// A [`Space`] measured in tiles: the runtime tile count per axis. `Space` names
/// the axes, `Grid` gives their size in tiles. Read by label
/// ([`tiles`](Grid::tiles)); a `Point` is a coordinate within a grid.
#[derive(CubeType)]
pub struct Grid {
    counts: Sequence<usize>,
    #[cube(comptime)]
    frame: Space,
}

#[cube]
impl Grid {
    /// Wrap per-axis tile `counts` (runtime, in `frame` order).
    pub fn new(counts: Sequence<usize>, #[comptime] frame: Space) -> Grid {
        Grid { counts, frame }
    }

    /// Tiles along `axis`, located via the grid's frame.
    pub fn tiles(&self, #[comptime] axis: Axis) -> usize {
        *self.counts.index(comptime!(self.frame.position(axis)))
    }

    /// The space these counts are over.
    pub fn frame(&self) -> comptime_type!(Space) {
        comptime!(self.frame.clone())
    }
}
