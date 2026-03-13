use cubecl;
use cubecl::prelude::*;

use cubek_std::tile::StridedTile;

use crate::components::tile::matmul::InnerMatmul;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct Query<IM: InnerMatmul> {
    pub fragment: IM::Lhs,
}

#[cube]
impl<IM: InnerMatmul> Query<IM> {
    pub fn new(#[comptime] config: IM::Config) -> Query<IM> {
        Query::<IM> {
            fragment: IM::allocate_lhs(config),
        }
    }

    /// Loads the query data into the fragment
    pub fn update<E: Numeric, ES: Size>(&mut self, tile: &StridedTile<E, ES>) {
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
            fragment: IM::allocate_rhs_transposed(config),
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
