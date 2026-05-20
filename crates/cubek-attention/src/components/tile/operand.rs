use cubecl;
use cubecl::prelude::*;

use cubek_std::tile::{Plane, Tile};

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct Query<L: Numeric> {
    pub tile: Tile<L, Plane>,
}

#[cube]
impl<L: Numeric> Query<L> {
    pub fn new(tile: Tile<L, Plane>) -> Query<L> {
        Query::<L> { tile }
    }
}

#[derive(CubeType)]
pub struct Key<R: Numeric> {
    pub tile: Tile<R, Plane>,
}

#[cube]
impl<R: Numeric> Key<R> {
    pub fn new(tile: Tile<R, Plane>) -> Key<R> {
        Key::<R> { tile }
    }
}

#[derive(CubeType)]
pub struct Value<R: Numeric> {
    pub tile: Tile<R, Plane>,
}

#[cube]
impl<R: Numeric> Value<R> {
    pub fn new(tile: Tile<R, Plane>) -> Value<R> {
        Value::<R> { tile }
    }
}
