//! 2-D [`Layout`]s presenting a tensor's storage to the leaf: one windows a tile
//! out of a tensor by its semantic origin, the other strides a flat smem tile.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand},
};

/// Re-presents one tile as a 2-D `[Tile0, Tile1]` view by offsetting local tile
/// coords to the tile's **semantic origin** `(row0, col0)`. It works in the
/// tensor's logical coordinates; whatever tiling the underlying view encodes (or
/// doesn't) is the view's own business, not this layout's.
#[derive(CubeType, Clone)]
pub struct TileWindow {
    row0: usize,
    col0: usize,
    tile_shape: Coords2d,
}

#[cube]
impl TileWindow {
    /// Window the tile whose origin is `(row0, col0)` in semantic coords, of size
    /// `rows × cols`.
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
