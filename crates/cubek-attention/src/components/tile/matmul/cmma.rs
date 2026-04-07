use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::matmul::InnerMatmul;

use cubek_std::{TileSize, tile::StridedTile};

#[derive(CubeType)]
pub struct CmmaMatmul<A: Numeric, B: Numeric, CD: Numeric> {
    #[cube(comptime)]
    _phantom: PhantomData<(A, B, CD)>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaMatmulConfig {
    pub tile_size: TileSize,
}

#[cube]
impl<A: Numeric, B: Numeric, CD: Numeric> InnerMatmul for CmmaMatmul<A, B, CD> {
    type Lhs = cmma::Matrix<A>;
    type Rhs = cmma::Matrix<B>;
    type Acc = cmma::Matrix<CD>;
    type Config = CmmaMatmulConfig;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        let size = config.tile_size;

        unsafe {
            cmma::Matrix::<A>::uninitialized(
                cmma::MatrixIdent::A,
                size.m() as usize,
                size.n() as usize,
                size.k() as usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
        let size = config.tile_size;
        unsafe {
            cmma::Matrix::<B>::uninitialized(
                cmma::MatrixIdent::B,
                size.m() as usize,
                size.n() as usize,
                size.k() as usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
    }

    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Self::Rhs {
        let size = config.tile_size;
        unsafe {
            cmma::Matrix::<B>::uninitialized(
                cmma::MatrixIdent::B,
                size.m() as usize,
                size.n() as usize,
                size.k() as usize,
                cmma::MatrixLayout::ColMajor,
            )
        }
    }

    fn load_lhs<E: Numeric, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Lhs) {
        let stride = tile.unvectorized_stride();
        let slice = tile.as_slice();
        cmma::load(fragment, &slice, stride);
    }

    fn load_rhs<E: Float, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Rhs) {
        let stride = tile.unvectorized_stride();
        let slice = tile.as_slice();
        cmma::load(fragment, &slice, stride);
    }

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Acc,
        #[comptime] _tile_size: TileSize,
    ) {
        cmma::execute::<A, B, CD, CD>(lhs, rhs, out, out);
    }
}
