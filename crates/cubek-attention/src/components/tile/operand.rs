use cubecl;
use cubecl::prelude::*;

<<<<<<< HEAD
use cubek_matmul::components::tile_matmul::{Plane, Tile};
=======
use cubek_matmul::components::tile::Tile;
>>>>>>> main

#[derive(CubeType)]
/// Query input to the Tile Attention
pub struct Query<L: Numeric, VL: Size> {
<<<<<<< HEAD
    pub tile: Tile<L, VL, Plane, ReadWrite>,
=======
    pub tile: Tile<L, VL, ReadWrite>,
>>>>>>> main
}

#[cube]
impl<L: Numeric, VL: Size> Query<L, VL> {
<<<<<<< HEAD
    pub fn new(tile: Tile<L, VL, Plane, ReadWrite>) -> Query<L, VL> {
=======
    pub fn new(tile: Tile<L, VL, ReadWrite>) -> Query<L, VL> {
>>>>>>> main
        Query::<L, VL> { tile }
    }
}

#[derive(CubeType)]
pub struct Key<R: Numeric, VR: Size> {
<<<<<<< HEAD
    pub tile: Tile<R, VR, Plane, ReadWrite>,
=======
    pub tile: Tile<R, VR, ReadWrite>,
>>>>>>> main
}

#[cube]
impl<R: Numeric, VR: Size> Key<R, VR> {
<<<<<<< HEAD
    pub fn new(tile: Tile<R, VR, Plane, ReadWrite>) -> Key<R, VR> {
=======
    pub fn new(tile: Tile<R, VR, ReadWrite>) -> Key<R, VR> {
>>>>>>> main
        Key::<R, VR> { tile }
    }
}

#[derive(CubeType)]
pub struct Value<R: Numeric, VR: Size> {
<<<<<<< HEAD
    pub tile: Tile<R, VR, Plane, ReadWrite>,
=======
    pub tile: Tile<R, VR, ReadWrite>,
>>>>>>> main
}

#[cube]
impl<R: Numeric, VR: Size> Value<R, VR> {
<<<<<<< HEAD
    pub fn new(tile: Tile<R, VR, Plane, ReadWrite>) -> Value<R, VR> {
=======
    pub fn new(tile: Tile<R, VR, ReadWrite>) -> Value<R, VR> {
>>>>>>> main
        Value::<R, VR> { tile }
    }
}
