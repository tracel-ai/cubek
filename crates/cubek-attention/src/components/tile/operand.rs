use cubecl;
use cubecl::prelude::*;

use cubek_matmul::components::tile::Tilex;

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct Query<L: Numeric, VL: Size> {
    pub tile: Tilex<L, VL, ReadWrite>,
}

#[cube]
impl<L: Numeric, VL: Size> Query<L, VL> {
    pub fn new(tile: Tilex<L, VL, ReadWrite>) -> Query<L, VL> {
        Query::<L, VL> { tile }
    }
}

#[derive(CubeType)]
pub struct Key<R: Numeric, VR: Size> {
    pub tile: Tilex<R, VR, ReadWrite>,
}

#[cube]
impl<R: Numeric, VR: Size> Key<R, VR> {
    pub fn new(tile: Tilex<R, VR, ReadWrite>) -> Key<R, VR> {
        Key::<R, VR> { tile }
    }
}

#[derive(CubeType)]
pub struct Value<R: Numeric, VR: Size> {
    pub tile: Tilex<R, VR, ReadWrite>,
}

#[cube]
impl<R: Numeric, VR: Size> Value<R, VR> {
    pub fn new(tile: Tilex<R, VR, ReadWrite>) -> Value<R, VR> {
        Value::<R, VR> { tile }
    }
}
