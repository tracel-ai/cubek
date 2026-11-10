use crate::linear_recurrence::conv::cube::PastTiles;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube};

/// Every operand of the conv as a tile.
#[derive(CubeType)]
pub struct QkNormedConvTiles {
    pub past: PastTiles,
    pub filters: Tile,
    pub query: Tile,
    pub key: Tile,
    pub value: Tile,
}

#[cube]
impl QkNormedConvTiles {
    pub fn at(&self, region: &Region) -> QkNormedConvTiles {
        QkNormedConvTiles {
            past: self.past.at(region),
            filters: self.filters.at(region),
            query: self.query.at(region),
            key: self.key.at(region),
            value: self.value.at(region),
        }
    }
}
