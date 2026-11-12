use super::past_taps::PastTiles;
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Every operand of the causal conv as a tile.
#[derive(CubeType)]
pub struct CausalConvTiles {
    pub past: PastTiles,
    pub filters: Tile,
    pub bias: ComptimeOption<Tile>,
    pub output: Tile,
}

#[cube]
impl CausalConvTiles {
    pub fn at(&self, region: &Region) -> CausalConvTiles {
        CausalConvTiles {
            past: self.past.at(region),
            filters: self.filters.at(region),
            bias: self.bias.clone().map(|bias| bias.at(region)),
            output: self.output.at(region),
        }
    }
}
