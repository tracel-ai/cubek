use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube};

/// Every operand of the conv as a tile.
#[derive(CubeType)]
pub struct GatedDeltaConvTiles {
    pub qkv: Tile,
    pub filters: Tile,
    /// The window store, placed at the row's slot.
    pub window: Tile,
    pub lengths: Tile,
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
            window: self.window.at(region),
            lengths: self.lengths.at(region),
            query: self.query.at(region),
            key: self.key.at(region),
            value: self.value.at(region),
            columns: self.columns,
        }
    }
}
