use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::matmul::InnerMatmul;
use crate::components::tile::pipeline::UnitTile;

use cubek_std::TileSize;
use cubek_std::tile::StridedTile;

#[derive(CubeType)]
pub struct UnitMatmul<A: Numeric, B: Numeric, CD: Numeric> {
    #[cube(comptime)]
    _phantom: PhantomData<(A, B, CD)>,
}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitMatmulConfig {}

#[cube]
impl<A: Numeric, B: Numeric, CD: Numeric> InnerMatmul for UnitMatmul<A, B, CD> {
    type Lhs = UnitTile<A>;
    type Rhs = UnitTile<B>;
    type Acc = UnitTile<CD>;
    type Config = UnitMatmulConfig;

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

    fn load_rhs_transposed<E: Float, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Rhs) {
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
