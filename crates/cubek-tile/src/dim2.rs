//! The 2-D leaf world. Everything keyed on `Coords2d` lives here, behind the
//! [`partition`](super::Tile::partition) seam — `row`/`col` and the fixed axis
//! pair `(0, 1)` are confined to this file. Holds the two leaf [`Layout`]s
//! ([`TileWindow`], [`SmemTileLayout`]), the element copy, and the `Coords2d`
//! half of [`Tile`](super::Tile).

use cubecl::{
    prelude::*,
    std::tensor::{
        ViewMut,
        layout::{Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand},
    },
};

// Glob brings sibling items *and* the cube-macro-generated `*Expand` companions.
use super::*;

/// Offsets local tile coords to the tile's origin `(row0, col0)` in the tensor's
/// logical coordinates.
#[derive(CubeType, Clone)]
pub struct TileWindow {
    row0: usize,
    col0: usize,
    tile_shape: Coords2d,
}

#[cube]
impl TileWindow {
    /// Window the `rows × cols` tile at origin `(row0, col0)`.
    pub fn new(row0: usize, col0: usize, #[comptime] rows: usize, #[comptime] cols: usize) -> Self {
        TileWindow {
            row0,
            col0,
            tile_shape: (
                u32::from_int(comptime!(rows as i64)),
                u32::from_int(comptime!(cols as i64)),
            ),
        }
    }
}

#[cube]
impl Layout for TileWindow {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (t0, t1) = pos;
        let mut out = CoordsDyn::new();
        out.push(self.row0 as u32 + t0);
        out.push(self.col0 as u32 + t1);
        out
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        let in_bounds = self.is_in_bounds(pos);
        (self.to_source_pos(pos), in_bounds)
    }

    fn shape(&self) -> Self::Coordinates {
        self.tile_shape
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (t0, t1) = pos;
        let (s0, s1) = self.tile_shape;
        t0 < s0 && t1 < s1
    }
}

/// Row-major 2-D layout over a flat smem buffer of `rows × cols`.
#[derive(CubeType, Clone)]
pub struct SmemTileLayout {
    shape: Coords2d,
    strides: Coords2d,
}

#[cube]
impl Layout for SmemTileLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = Coords1d;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (t0, t1) = pos;
        let (s0, s1) = self.strides;
        (t0 * s0 + t1 * s1) as usize
    }

    fn to_source_pos_checked(&self, pos: Self::Coordinates) -> (Self::SourceCoordinates, bool) {
        (self.to_source_pos(pos), self.is_in_bounds(pos))
    }

    fn shape(&self) -> Self::Coordinates {
        self.shape
    }

    fn is_in_bounds(&self, pos: Self::Coordinates) -> bool {
        let (t0, t1) = pos;
        let (s0, s1) = self.shape;
        t0 < s0 && t1 < s1
    }
}

#[cube]
pub fn smem_tile_layout(#[comptime] rows: usize, #[comptime] cols: usize) -> SmemTileLayout {
    SmemTileLayout {
        shape: (
            u32::from_int(comptime!(rows as i64)),
            u32::from_int(comptime!(cols as i64)),
        ),
        strides: (u32::from_int(comptime!(cols as i64)), u32::from_int(1)),
    }
}

#[cube]
impl<'a, E: Numeric, S: Size> Tile<'a, E, S, Coords2d> {
    /// Copy `src` into this tile — a gmem↔smem element copy.
    pub fn copy_from(&mut self, src: &Tile<'_, E, S, Coords2d>) {
        copy_2d::<E, S>(&mut self.view, &src.view);
    }
}

/// Wrap a shared-memory view as a [`Smem`](TileKind::Smem) tile.
#[cube]
pub fn stage_smem<'a, E: Numeric, S: Size>(
    view: ViewMut<'a, Vector<E, S>, Coords2d>,
    #[comptime] space: Space,
    partitioner: Partitioner,
) -> Tile<'a, E, S, Coords2d> {
    Tile::<'a, E, S, Coords2d> {
        view,
        partitioner,
        space,
        kind: comptime!(TileKind::Smem),
    }
}

/// Element-wise copy of `src` into `dst` (same 2-D shape).
#[cube]
pub fn copy_2d<E: Numeric, S: Size>(
    dst: &mut ViewMut<'_, Vector<E, S>, Coords2d>,
    src: &ViewMut<'_, Vector<E, S>, Coords2d>,
) {
    let (h, w) = src.shape();
    for i in 0..h {
        for j in 0..w {
            dst.write((i, j), src.read((i, j)));
        }
    }
}
