//! The **tile**: a view onto data, tagged with the semantic [`Space`] it lives
//! in and carrying its [`Partitioner`]. This is the launchable unit — kernels
//! take tiles directly, and a tile *is* a semantic view (the physical tiling, if
//! any, lives in the view's layout).
//!
//! Generic over the view's coordinate type: `CoordsDyn` is a whole tensor (a
//! launch input or the accumulator), `Coords2d` a leaf or shared-memory tile. IO
//! is fixed `ReadWrite` — the accumulator needs it, and read operands are simply
//! never written. A comptime [`TileKind`] tags the memory space.

use cubecl::{
    prelude::*,
    std::tensor::{
        View,
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
pub struct Tile<E: Numeric, S: Size, C: Coordinates> {
    pub(crate) view: View<Vector<E, S>, C, ReadWrite>,
    pub(crate) partitioner: Partitioner,
    #[cube(comptime)]
    pub(crate) space: Space,
    #[cube(comptime)]
    pub(crate) kind: TileKind,
}

// Whole-tensor tiles (`CoordsDyn`): partition into leaves, accumulate (the walk),
// and report their tile counts. These are the launchable operands.
#[cube]
impl<E: Numeric, S: Size> Tile<E, S, CoordsDyn> {
    /// Select the sub-tile at grid `point`, windowed to its semantic origin. Each
    /// tile reads the point along its *own* axes, so operands and accumulator at
    /// the same point return matching sub-tiles even in different spaces.
    pub fn partition(&self, point: &Point) -> Tile<E, S, Coords2d> {
        let g0 = point.get(comptime!(self.space.axis_at(0)));
        let g1 = point.get(comptime!(self.space.axis_at(1)));
        let rows = self.partitioner.sub_tile_edge(comptime!(self.space.axis_at(0)));
        let cols = self.partitioner.sub_tile_edge(comptime!(self.space.axis_at(1)));
        // The tile's origin in semantic coords: its grid index times its size.
        let layout = TileWindow::new(g0 * rows, g1 * cols, rows, cols);
        Tile::<E, S, Coords2d> {
            view: self.view.view_mut(layout),
            partitioner: self.partitioner.clone(),
            space: comptime!(self.space.clone()),
            kind: comptime!(TileKind::GmemLeaf),
        }
    }

    /// Accumulate `lhs · rhs` into this whole-tensor accumulator (the gmem walk).
    pub fn mma(&self, lhs: &Tile<E, S, CoordsDyn>, rhs: &Tile<E, S, CoordsDyn>) {
        mma_gmem::<E, S>(self, lhs, rhs);
    }

    /// Runtime number of tiles along `axis`: the semantic extent over that axis
    /// divided by the sub-tile size.
    pub fn tiles(&self, #[comptime] axis: Axis) -> u32 {
        let shape = self.view.shape();
        let extent = *shape.index(comptime!(self.space.position(axis)));
        extent / self.partitioner.sub_tile_edge(axis)
    }
}

// Leaf / shared-memory tiles (`Coords2d`): the scalar accumulate and the copies.
#[cube]
impl<E: Numeric, S: Size> Tile<E, S, Coords2d> {
    /// Scalar tile multiply-accumulate into this leaf accumulator.
    pub fn mma(&self, lhs: &Tile<E, S, Coords2d>, rhs: &Tile<E, S, Coords2d>) {
        mma_smem::<E, S>(&self.view, &lhs.view, &rhs.view);
    }

    /// Copy `src` into this tile — a gmem→smem load or an smem→gmem store; both
    /// are element copies once the view abstracts the backing memory.
    pub fn copy_from(&self, src: &Tile<E, S, Coords2d>) {
        copy_2d::<E, S>(&self.view, &src.view);
    }
}

/// Wrap a shared-memory view as a [`Smem`](TileKind::Smem) tile.
#[cube]
pub fn stage_smem<E: Numeric, S: Size>(
    view: View<Vector<E, S>, Coords2d, ReadWrite>,
    #[comptime] space: Space,
    partitioner: Partitioner,
) -> Tile<E, S, Coords2d> {
    Tile::<E, S, Coords2d> {
        view,
        partitioner,
        space,
        kind: comptime!(TileKind::Smem),
    }
}
