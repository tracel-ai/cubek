//! The [`Tile`]: a view tagged with its [`Space`] and [`Partitioner`]. The
//! launchable unit — kernels take tiles directly.
//!
//! The arity-agnostic half: `CoordsDyn` whole tensors and the
//! [`partition`](Tile::partition) seam that collapses them into `Coords2d`
//! leaves. The 2-D leaf machinery lives in [`dim2`](super::dim2).

use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut,
        layout::{Coordinates, Coords2d, CoordsDyn},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// What memory a tile lives in.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum TileKind {
    /// A whole global tensor — a launch input or the accumulator.
    GmemWhole,
    /// One tile of a global tensor — a `partition` leaf.
    GmemLeaf,
    /// A shared-memory tile — staging buffer or leaf accumulator.
    Smem,
}

/// A view + its [`Space`] + [`Partitioner`]. Launchable; generic over the view's
/// coordinate type `C`.
#[derive(CubeType, CubeLaunch)]
pub struct Tile<'a, E: Numeric, S: Size, C: Coordinates + 'a> {
    pub view: ViewMut<'a, Vector<E, S>, C>,
    pub partitioner: Partitioner,
    #[cube(comptime)]
    pub space: Space,
    #[cube(comptime)]
    pub kind: TileKind,
}

#[cube]
impl<'a, E: Numeric, S: Size> Tile<'a, E, S, CoordsDyn> {
    /// The sub-tile at grid `point`, windowed to its origin. Each tile reads the
    /// point along its *own* axes, so operands and accumulator match even in
    /// different spaces.
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

    /// Number of tiles along `axis`: extent / sub-tile size.
    pub fn tiles(&self, #[comptime] axis: Axis) -> usize {
        let shape = self.view.shape();
        let extent = *shape.index(comptime!(self.space.position(axis))) as usize;
        extent / self.partitioner.sub_tile_edge(axis)
    }
}
