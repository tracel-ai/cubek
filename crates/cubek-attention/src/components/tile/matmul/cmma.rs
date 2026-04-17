use cubecl;
use cubecl::prelude::*;
use cubek_matmul::{
    components::tile::{Tilex, cmma_allocate_lhs, cmma_allocate_rhs, tilex_execute, tilex_load},
    definition::StageIdent,
};

use crate::components::tile::matmul::InnerMatmul;

use cubek_std::{MatrixLayout, TileSize};

#[derive(CubeType)]
pub struct CmmaMatmul {}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct CmmaMatmulConfig {
    pub tile_size: TileSize,
}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    InnerMatmul<L, VL, R, VR, A, VA> for CmmaMatmul
{
    type Config = CmmaMatmulConfig;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Tilex<L, VL, ReadWrite> {
        cmma_allocate_lhs(MatrixLayout::RowMajor, config.tile_size)
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Tilex<R, VR, ReadWrite> {
        cmma_allocate_rhs(MatrixLayout::RowMajor, config.tile_size)
    }

    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Tilex<R, VR, ReadWrite> {
        cmma_allocate_rhs(MatrixLayout::ColMajor, config.tile_size)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        source: &Tilex<E, ES, ReadOnly>,
        dest: &mut Tilex<L, VL, ReadWrite>,
    ) {
        tilex_load::<E, ES, L, VL, L, R, A>(source, dest, StageIdent::Lhs);
    }

    fn load_rhs<E: Float, ES: Size>(
        source: &Tilex<E, ES, ReadOnly>,
        dest: &mut Tilex<R, VR, ReadWrite>,
    ) {
        tilex_load::<E, ES, R, VR, L, R, A>(source, dest, StageIdent::Rhs);
    }

    fn execute(
        lhs: &Tilex<L, VL, ReadWrite>,
        rhs: &Tilex<R, VR, ReadWrite>,
        acc: &mut Tilex<A, VA, ReadWrite>,
        #[comptime] _tile_size: TileSize,
    ) {
        tilex_execute::<L, VL, R, VR, A, VA>(lhs, rhs, acc)
    }
}
