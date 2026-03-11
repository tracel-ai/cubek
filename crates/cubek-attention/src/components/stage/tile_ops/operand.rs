use cubecl;
use cubecl::prelude::*;

use crate::components::softmax::InnerMatmul;
use cubek_std::tile::StridedTile;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct QueryTile<IM: InnerMatmul> {
    pub fragment: IM::Lhs,
}

#[cube]
impl<IM: InnerMatmul> QueryTile<IM> {
    pub fn new(#[comptime] config: IM::Config) -> QueryTile<IM> {
        QueryTile::<IM> {
            fragment: IM::allocate_lhs(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update<E: Numeric>(&mut self, tile: &StridedTile<E>) {
        IM::load_lhs(tile, &mut self.fragment)
    }
}

#[derive(CubeType)]
pub struct Key<IM: InnerMatmul> {
    pub fragment: IM::Rhs,
}

#[cube]
impl<IM: InnerMatmul> Key<IM> {
    pub fn new(#[comptime] config: IM::Config) -> Self {
        Key::<IM> {
            fragment: IM::allocate_rhs(config),
        }
    }
}

#[derive(CubeType)]
pub struct Value<IM: InnerMatmul> {
    pub fragment: IM::Rhs,
}

#[cube]
impl<IM: InnerMatmul> Value<IM> {
    pub fn new(#[comptime] config: IM::Config) -> Self {
        Value::<IM> {
            fragment: IM::allocate_rhs(config),
        }
    }
}
