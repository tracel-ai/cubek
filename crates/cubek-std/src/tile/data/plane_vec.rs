use cubecl::{define_size, prelude::*};

use crate::{
    MatrixLayout, SwizzleModes, TileSize,
    tile::{Tile, TileScope},
};

// plane_vec_mat's fragment inner vector size (= reduce_vector_size). Bound at
// allocate time via `scope.register_size::<NPlaneVec>(reduce_vector_size)`.
// Decoupled from the outer enum `V` so the fragment is sized by the tile impl's
// needs, not the stage's vector size.
define_size!(pub NPlaneVec);

#[derive(CubeType)]
pub struct PlaneVecTile<N: Numeric> {
    // Fragment inner size is `NPlaneVec` (= reduce_vector_size).
    pub data: Array<Vector<N, NPlaneVec>>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub config: PlaneVecMatInnerProduct,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct PlaneVecMatInnerProduct {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
    pub reduce_vector_size: u32,
}

impl PlaneVecMatInnerProduct {
    pub fn new(
        tile_size: TileSize,
        plane_dim: u32,
        swizzle_modes: SwizzleModes,
        reduce_vector_size: u32,
    ) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
            reduce_vector_size,
        }
    }
}

// Binds the plane_vec_mat fragment's inner vector size (`NPlaneVec`) to the
// `reduce_vector_size` chosen by the tile config at allocation time.
#[cube]
#[allow(unused_variables)]
fn register_reduce_vector_size(#[comptime] reduce_vector_size: u32) {
    intrinsic!(|scope| {
        scope.register_size::<NPlaneVec>(reduce_vector_size as usize);
    });
}

#[cube]
pub fn planevec_allocate_lhs<L: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: PlaneVecMatInnerProduct,
) -> Tile<L, Sc, ReadWrite> {
    register_reduce_vector_size(config.reduce_vector_size);
    Tile::new_PlaneVec(PlaneVecTile::<L> {
        data: Array::new(1usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn planevec_allocate_rhs<R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: PlaneVecMatInnerProduct,
) -> Tile<R, Sc, ReadWrite> {
    register_reduce_vector_size(config.reduce_vector_size);
    Tile::new_PlaneVec(PlaneVecTile::<R> {
        data: Array::new(config.tile_size.n() as usize),
        matrix_layout: layout,
        config,
    })
}

#[cube]
pub fn planevec_allocate_acc<A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] config: PlaneVecMatInnerProduct,
) -> Tile<A, Sc, ReadWrite> {
    register_reduce_vector_size(config.reduce_vector_size);
    Tile::new_PlaneVec(PlaneVecTile::<A> {
        data: Array::new(config.tile_size.n() as usize),
        matrix_layout: layout,
        config,
    })
}
