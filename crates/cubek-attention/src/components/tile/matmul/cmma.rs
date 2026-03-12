use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::matmul::InnerMatmul;

use cubek_std::TileSize;
use cubek_std::tile::StridedTile;

#[derive(CubeType)]
pub struct CmmaMatmul<A: Numeric, B: Numeric, CD: Numeric> {
    #[cube(comptime)]
    _phantom: PhantomData<(A, B, CD)>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaMatmulConfig {}

#[cube]
impl<A: Numeric, B: Numeric, CD: Numeric> InnerMatmul for CmmaMatmul<A, B, CD> {
    type Lhs = cmma::Matrix<A>;
    type Rhs = cmma::Matrix<B>;
    type Acc = cmma::Matrix<CD>;
    type Config = CmmaMatmulConfig;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs {
        todo!()
    }

    fn load_lhs<E: Numeric, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Lhs) {
        todo!()
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs {
        todo!()
    }

    fn load_rhs_plain<E: Float, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Rhs) {
        todo!()
    }

    fn load_rhs_transposed<E: Float, ES: Size>(
        tile: &StridedTile<E, ES>,
        fragment: &mut Self::Rhs,
    ) {
        todo!()
    }

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Acc,
        #[comptime] tile_size: TileSize,
    ) {
        todo!()
    }
}
