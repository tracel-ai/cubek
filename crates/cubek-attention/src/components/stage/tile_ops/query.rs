use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::softmax::InnerMatmul;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct QueryTile<SMM: InnerMatmul> {
    pub fragment: SMM::Lhs,
}

#[cube]
impl<SMM: InnerMatmul> QueryTile<SMM> {
    pub fn new(#[comptime] config: SMM::Config) -> QueryTile<SMM> {
        QueryTile::<SMM> {
            fragment: SMM::allocate_lhs(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update<E: Numeric>(&mut self, tile: &StridedTile<E>) {
        SMM::load_lhs(tile, &mut self.fragment)
    }
}
