use cubecl;
use cubecl::prelude::*;

use cubek_std::tile::{Plane, Tile};

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct Query<L: Numeric> {
    pub tile: Tile<L, Plane, ReadWrite>,
}

#[cube]
impl<L: Numeric> Query<L> {
    pub fn new(tile: Tile<L, Plane, ReadWrite>) -> Query<L> {
        Query::<L> { tile }
    }
}

#[derive(CubeType)]
pub struct Key<R: Numeric> {
    pub tile: Tile<R, Plane, ReadWrite>,
}

#[cube]
impl<R: Numeric> Key<R> {
    pub fn new(tile: Tile<R, Plane, ReadWrite>) -> Key<R> {
        Key::<R> { tile }
    }
}

#[derive(CubeType)]
pub struct Value<R: Numeric> {
    pub tile: Tile<R, Plane, ReadWrite>,
}

#[cube]
impl<R: Numeric> Value<R> {
    pub fn new(tile: Tile<R, Plane, ReadWrite>) -> Value<R> {
        Value::<R> { tile }
    }
}
