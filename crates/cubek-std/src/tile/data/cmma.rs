use cubecl::{
    cmma::Matrix as CubeMatrix,
    cmma::{self},
    prelude::*,
};

use crate::{
    MatrixLayout, SwizzleModes, TileSize, as_cmma_layout,
    tile::{Tile, TileScope},
};

#[derive(CubeType)]
pub struct CmmaTile<N: Numeric> {
    pub matrix: CubeMatrix<N>,
    #[cube(comptime)]
    pub matrix_layout: MatrixLayout,
    #[cube(comptime)]
    pub tile_size: TileSize,
}

#[derive(Copy, Clone, Eq, PartialEq, Hash, Debug)]
pub struct CmmaMatmul {
    pub tile_size: TileSize,
    pub plane_dim: u32,
    pub swizzle_modes: SwizzleModes,
}

impl CmmaMatmul {
    pub fn new(tile_size: TileSize, plane_dim: u32, swizzle_modes: SwizzleModes) -> Self {
        Self {
            tile_size,
            plane_dim,
            swizzle_modes,
        }
    }
}

#[cube]
pub fn cmma_allocate_lhs<L: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<L, Sc, ReadWrite> {
    let fragment = unsafe {
        cmma::Matrix::<L>::uninitialized(
            cmma::MatrixIdent::A,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            as_cmma_layout(layout),
        )
    };
    Tile::new_Cmma(CmmaTile::<L> {
        matrix: fragment,
        matrix_layout: layout,
        tile_size,
    })
}

#[cube]
pub fn cmma_allocate_rhs<R: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<R, Sc, ReadWrite> {
    let fragment = unsafe {
        cmma::Matrix::<R>::uninitialized(
            cmma::MatrixIdent::B,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            as_cmma_layout(layout),
        )
    };
    Tile::new_Cmma(CmmaTile::<R> {
        matrix: fragment,
        matrix_layout: layout,
        tile_size,
    })
}

#[cube]
pub fn cmma_allocate_acc<A: Numeric, Sc: TileScope>(
    #[comptime] layout: MatrixLayout,
    #[comptime] tile_size: TileSize,
) -> Tile<A, Sc, ReadWrite> {
    let fragment = unsafe {
        cmma::Matrix::<A>::uninitialized(
            cmma::MatrixIdent::Accumulator,
            tile_size.m as usize,
            tile_size.n as usize,
            tile_size.k as usize,
            cmma::MatrixLayout::Undefined,
        )
    };
    Tile::new_Cmma(CmmaTile::<A> {
        matrix: fragment,
        matrix_layout: layout,
        tile_size,
    })
}
