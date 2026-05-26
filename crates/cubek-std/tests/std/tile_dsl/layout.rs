//! 2-D [`Layout`]s that re-present a tiled tensor's storage to the leaf: one
//! pins a tile out of the physical 4-D grid, the other strides a flat smem tile.

use cubecl::{
    prelude::*,
    std::tensor::layout::{Coords1d, Coords2d, CoordsDyn, Layout, LayoutExpand},
};

/// Re-presents one tile of a tiled tensor (physical 4-D coords
/// `[Grid0, Grid1, Tile0, Tile1]`) as a 2-D `[Tile0, Tile1]` view by pinning the
/// grid coordinates.
#[derive(CubeType, Clone)]
pub struct TileSelectLayout {
    g0: u32,
    g1: u32,
    tile_shape: Coords2d,
}

#[cube]
impl TileSelectLayout {
    pub fn new(g0: u32, g1: u32, #[comptime] rows: u32, #[comptime] cols: u32) -> Self {
        TileSelectLayout {
            g0,
            g1,
            tile_shape: (
                u32::from_int(comptime!(rows as i64)),
                u32::from_int(comptime!(cols as i64)),
            ),
        }
    }
}

#[cube]
impl Layout for TileSelectLayout {
    type Coordinates = Coords2d;
    type SourceCoordinates = CoordsDyn;

    fn to_source_pos(&self, pos: Self::Coordinates) -> Self::SourceCoordinates {
        let (t0, t1) = pos;
        let mut out = CoordsDyn::new();
        out.push(self.g0);
        out.push(self.g1);
        out.push(t0);
        out.push(t1);
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
pub fn smem_tile_layout(#[comptime] rows: u32, #[comptime] cols: u32) -> SmemTileLayout {
    SmemTileLayout {
        shape: (
            u32::from_int(comptime!(rows as i64)),
            u32::from_int(comptime!(cols as i64)),
        ),
        strides: (u32::from_int(comptime!(cols as i64)), u32::from_int(1)),
    }
}
