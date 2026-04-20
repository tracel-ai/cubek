use cubecl;
use cubecl::prelude::*;

use cubek_matmul::components::tile_matmul::{Plane, Tile};

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct Query<L: Numeric, VL: Size> {
    pub tile: Tile<L, VL, Plane, ReadWrite>,
}

#[cube]
impl<L: Numeric, VL: Size> Query<L, VL> {
    pub fn new(tile: Tile<L, VL, Plane, ReadWrite>) -> Query<L, VL> {
        Query::<L, VL> { tile }
    }
}

#[derive(CubeType)]
pub struct Key<R: Numeric, VR: Size> {
    pub tile: Tile<R, VR, Plane, ReadWrite>,
}

#[cube]
impl<R: Numeric, VR: Size> Key<R, VR> {
    pub fn new(tile: Tile<R, VR, Plane, ReadWrite>) -> Key<R, VR> {
        Key::<R, VR> { tile }
    }
}

#[derive(CubeType)]
pub struct Value<R: Numeric, VR: Size> {
    pub tile: Tile<R, VR, Plane, ReadWrite>,
}

#[cube]
impl<R: Numeric, VR: Size> Value<R, VR> {
    pub fn new(tile: Tile<R, VR, Plane, ReadWrite>) -> Value<R, VR> {
        Value::<R, VR> { tile }
    }
}
