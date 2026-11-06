use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Every operand of the conv as a tile. The window and the lengths are absent over a stream.
#[derive(CubeType)]
pub struct GatedDeltaConvTiles {
    /// The row's newest values over a window, the sequence's whole stream, its context columns
    /// first, over a stream.
    pub qkv: Tile,
    pub filters: Tile,
    /// The window store, placed at the row's slot.
    pub window: ComptimeOption<Tile>,
    pub lengths: ComptimeOption<Tile>,
    pub query: Tile,
    pub key: Tile,
    pub value: Tile,
    /// The ring's columns, the one extent of the window no tile carries.
    pub columns: usize,
}

#[cube]
impl GatedDeltaConvTiles {
    pub fn at(&self, region: &Region) -> GatedDeltaConvTiles {
        GatedDeltaConvTiles {
            qkv: self.qkv.at(region),
            filters: self.filters.at(region),
            window: self.window.clone().map(|window| window.at(region)),
            lengths: self.lengths.clone().map(|lengths| lengths.at(region)),
            query: self.query.at(region),
            key: self.key.at(region),
            value: self.value.at(region),
            columns: self.columns,
        }
    }
}
