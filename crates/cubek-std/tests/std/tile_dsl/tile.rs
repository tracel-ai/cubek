//! The **tile**: its memory-space kinds ([`TileKind`]), the three verbs
//! ([`Tile::partition`] / [`Tile::mma`] / [`Tile::copy_from`]), and the
//! [`Gmem`]/[`Smem`] factories that build them.

use cubecl::{
    prelude::*,
    std::tensor::{
        View,
        layout::{Coords2d, CoordsDyn},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// A view onto a data source, tagged with the [`Space`] it lives in (its own
/// axes + extents, e.g. matmul's `lhs ∈ {M, K}`) and the [`Partitioner`].
/// [`TileKind`] is the memory space the three verbs dispatch on.
#[derive(CubeType)]
pub struct Tile<E: Numeric, S: Size> {
    pub kind: TileKind<E, S>,
    #[cube(comptime)]
    pub(crate) space: Space,
    pub(crate) partitioner: Partitioner,
}

#[derive(CubeType)]
#[allow(dead_code, non_snake_case)]
pub enum TileKind<E: Numeric, S: Size> {
    /// Whole tiled global tensor, read-only — a `partition` source (operand).
    GmemTiledRead(View<Vector<E, S>, CoordsDyn>),
    /// Whole tiled global tensor, writable — the accumulator.
    GmemTiledWrite(View<Vector<E, S>, CoordsDyn, ReadWrite>),
    /// One tile, read-only — a `partition` leaf, source of a gmem→smem load.
    GmemRead(View<Vector<E, S>, Coords2d>),
    /// One tile, writable — a `partition` leaf, destination of a writeback.
    GmemWrite(View<Vector<E, S>, Coords2d, ReadWrite>),
    /// One shared-memory tile (RW) — staging buffer / leaf accumulator.
    Smem(View<Vector<E, S>, Coords2d, ReadWrite>),
}

// The tile API: exactly three verbs.
#[cube]
impl<E: Numeric, S: Size> Tile<E, S> {
    /// Select the sub-tile at grid `point`. Each tile reads the point along its
    /// own axes, so operands and accumulator at the same point return matching
    /// sub-tiles even though they live in different spaces.
    pub fn partition(&self, point: &Point) -> Tile<E, S> {
        let g0 = point.get(comptime!(self.space.axis_at(0)));
        let g1 = point.get(comptime!(self.space.axis_at(1)));
        let rows = self.partitioner.sub_tile_edge(comptime!(self.space.axis_at(0)));
        let cols = self.partitioner.sub_tile_edge(comptime!(self.space.axis_at(1)));
        // The tile's origin in semantic coords: its grid index times its size.
        let layout = TileWindow::new(g0 * rows, g1 * cols, rows, cols);
        match &self.kind {
            TileKind::GmemTiledRead(grid) => Gmem::read_leaf::<E, S>(
                grid.view(layout),
                comptime!(self.space.clone()),
                self.partitioner.clone(),
            ),
            TileKind::GmemTiledWrite(grid) => Gmem::write_leaf::<E, S>(
                grid.view_mut(layout),
                comptime!(self.space.clone()),
                self.partitioner.clone(),
            ),
            TileKind::GmemRead(_) | TileKind::GmemWrite(_) | TileKind::Smem(_) => {
                panic!("Tile::partition: only a whole tiled tensor can be partitioned")
            }
        }
    }

    /// Register a multiply-accumulate with `self` as the accumulator. Picks a
    /// strategy keyed on the accumulator's memory space and recurses to the leaf.
    pub fn mma(&self, lhs: &Tile<E, S>, rhs: &Tile<E, S>) {
        match &self.kind {
            TileKind::GmemTiledWrite(_) => mma_gmem::<E, S>(self, lhs, rhs),
            TileKind::GmemWrite(_) | TileKind::Smem(_) => mma_leaf::<E, S>(self, lhs, rhs),
            TileKind::GmemTiledRead(_) | TileKind::GmemRead(_) => {
                panic!("Tile::mma: no strategy for this accumulator")
            }
        }
    }

    /// Register a copy into `self`; the destination's memory space decides the
    /// lowering (smem ← gmem load, or gmem ← smem writeback).
    pub fn copy_from(&self, src: &Tile<E, S>) {
        match &self.kind {
            TileKind::Smem(dst) => smem_load_from::<E, S>(dst, src),
            TileKind::GmemWrite(dst) => gmem_store_from::<E, S>(dst, src),
            TileKind::GmemTiledRead(_) | TileKind::GmemTiledWrite(_) | TileKind::GmemRead(_) => {
                panic!("Tile::copy_from: destination cannot be written into")
            }
        }
    }

    /// The runtime number of tiles along `axis`: the tensor's semantic extent on
    /// that axis divided by the sub-tile size. Read from the view's *semantic*
    /// shape, so it's the same whether or not the storage is physically tiled.
    pub fn tiles(&self, #[comptime] axis: Axis) -> u32 {
        let shape = self.semantic_shape();
        let extent = *shape.index(comptime!(self.space.position(axis)));
        extent / self.partitioner.sub_tile_edge(axis)
    }

    /// The semantic (logical) shape of a whole tensor view (panics for
    /// leaves/smem).
    fn semantic_shape(&self) -> CoordsDyn {
        match &self.kind {
            TileKind::GmemTiledRead(v) => v.shape(),
            TileKind::GmemTiledWrite(v) => v.shape(),
            TileKind::GmemRead(_) | TileKind::GmemWrite(_) | TileKind::Smem(_) => {
                panic!("Tile::tiles: only a whole tensor has a tile grid")
            }
        }
    }
}

/// Factory for global-memory tiles. The client supplies the [`Space`] each
/// operand lives in; the engine has no notion of "lhs"/"out".
pub struct Gmem;

#[cube]
impl Gmem {
    /// A whole tiled tensor, read-only — a `partition` source.
    pub fn tiled_read<E: Numeric, S: Size>(
        view: View<Vector<E, S>, CoordsDyn>,
        #[comptime] space: Space,
        partitioner: Partitioner,
    ) -> Tile<E, S> {
        Tile::<E, S> {
            kind: TileKind::new_GmemTiledRead(view),
            space,
            partitioner,
        }
    }

    /// A whole tiled tensor, writable — the accumulator.
    pub fn tiled_write<E: Numeric, S: Size>(
        view: View<Vector<E, S>, CoordsDyn, ReadWrite>,
        #[comptime] space: Space,
        partitioner: Partitioner,
    ) -> Tile<E, S> {
        Tile::<E, S> {
            kind: TileKind::new_GmemTiledWrite(view),
            space,
            partitioner,
        }
    }

    /// Read-only `partition` leaf — source of a gmem→smem load.
    pub fn read_leaf<E: Numeric, S: Size>(
        view: View<Vector<E, S>, Coords2d>,
        #[comptime] space: Space,
        partitioner: Partitioner,
    ) -> Tile<E, S> {
        Tile::<E, S> {
            kind: TileKind::new_GmemRead(view),
            space,
            partitioner,
        }
    }

    /// Writable `partition` leaf — destination of an smem→gmem writeback.
    pub fn write_leaf<E: Numeric, S: Size>(
        view: View<Vector<E, S>, Coords2d, ReadWrite>,
        #[comptime] space: Space,
        partitioner: Partitioner,
    ) -> Tile<E, S> {
        Tile::<E, S> {
            kind: TileKind::new_GmemWrite(view),
            space,
            partitioner,
        }
    }
}

/// Factory for shared-memory tiles (staging buffers / leaf accumulator).
pub struct Smem;

#[cube]
impl Smem {
    pub fn stage<E: Numeric, S: Size>(
        view: View<Vector<E, S>, Coords2d, ReadWrite>,
        #[comptime] space: Space,
        partitioner: Partitioner,
    ) -> Tile<E, S> {
        Tile::<E, S> {
            kind: TileKind::new_Smem(view),
            space,
            partitioner,
        }
    }
}
