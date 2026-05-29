//! The **tile**: a view onto data, tagged with the semantic [`Space`] it lives
//! in and carrying its [`Partitioner`]. This is the launchable unit — kernels
//! take tiles directly, and a tile *is* a semantic view (the physical tiling, if
//! any, lives in the view's layout).
//!
//! Generic over the view's coordinate type: `CoordsDyn` is a whole tensor (a
//! launch input or the accumulator), `Coords2d` a leaf or shared-memory tile. The
//! view is a [`ViewMut`] — the accumulator writes through it, and read operands
//! are simply never written. A comptime [`TileKind`] tags the memory space.
//!
//! This file is the **arity-agnostic** half: the [`Tile`] struct, [`TileKind`],
//! and the `CoordsDyn` whole-tensor impl. [`partition`](Tile::partition) is the
//! seam — it collapses an N-d tile into a `Coords2d` leaf; everything 2-D it
//! produces (the layouts, the copies) lives in [`dim2`](super::dim2).

use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut,
        layout::{Coordinates, Coords2d, CoordsDyn},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// What memory a tile lives in. The coordinate type already says whole
/// (`CoordsDyn`) vs leaf (`Coords2d`); this records the memory space.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TileKind {
    /// A whole tensor in global memory — a launch input or the accumulator.
    GmemWhole,
    /// One tile of a global tensor — a `partition` leaf.
    GmemLeaf,
    /// A shared-memory tile — staging buffer or leaf accumulator.
    Smem,
}

/// A view onto data + the semantic [`Space`] it lives in + its [`Partitioner`].
/// Launchable (`view` is a launch arg, `partitioner` rides along), so kernels
/// receive tiles directly. Generic over the view's coordinate type `C`.
#[derive(CubeType, CubeLaunch)]
pub struct Tile<'a, E: Numeric, S: Size, C: Coordinates + 'a> {
    pub view: ViewMut<'a, Vector<E, S>, C>,
    pub partitioner: Partitioner,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub kind: TileKind,
}

// Whole-tensor tiles (`CoordsDyn`): partition into leaves, accumulate (the walk),
// and report their tile counts. These are the launchable operands.
#[cube]
impl<'a, E: Numeric, S: Size> Tile<'a, E, S, CoordsDyn> {
    /// Select the sub-tile at grid `point`, windowed to its semantic origin. Each
    /// tile reads the point along its *own* axes, so operands and accumulator at
    /// the same point return matching sub-tiles even in different spaces.
    pub fn partition(&self, point: &Point) -> Tile<'a, E, S, Coords2d> {
        let g0 = point.get(comptime!(self.space.axis_at(0)));
        let g1 = point.get(comptime!(self.space.axis_at(1)));
        let rows = self
            .partitioner
            .sub_tile_edge(comptime!(self.space.axis_at(0)));
        let cols = self
            .partitioner
            .sub_tile_edge(comptime!(self.space.axis_at(1)));
        // The tile's origin in semantic coords: its grid index times its size.
        let layout = TileWindow::new(g0 * rows, g1 * cols, rows, cols);
        Tile::<'a, E, S, Coords2d> {
            view: self.view.clone().view_mut(layout),
            partitioner: self.partitioner.clone(),
            space: comptime!(self.space.clone()),
            kind: comptime!(TileKind::GmemLeaf),
        }
    }

    /// Runtime number of tiles along `axis`: the semantic extent over that axis
    /// divided by the sub-tile size.
    pub fn tiles(&self, #[comptime] axis: Axis) -> usize {
        let shape = self.view.shape();
        let extent = *shape.index(comptime!(self.space.position(axis))) as usize;
        extent / self.partitioner.sub_tile_edge(axis)
    }
}
