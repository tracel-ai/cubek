use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::definition::AttentionPrecision;
use crate::definition::attention_types::QG;
use crate::{components::tile::TileAttention, definition::attention_types::QGS};

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct QueryTile<AP: AttentionPrecision, TA: TileAttention<AP>> {
    pub fragment: TA::Query,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> QueryTile<AP, TA> {
    pub fn new(#[comptime] config: TA::Config) -> QueryTile<AP, TA> {
        QueryTile::<AP, TA> {
            fragment: TA::allocate_query(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update(&mut self, tile: &StridedTile<QG<AP>, QGS<AP>>) {
        TA::load_query(tile, &mut self.fragment)
    }
}
