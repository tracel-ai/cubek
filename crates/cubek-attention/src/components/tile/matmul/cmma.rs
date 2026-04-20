use cubecl;
use cubecl::prelude::*;
use cubek_matmul::{
    components::tile::{Tile, cmma_allocate_lhs, cmma_allocate_rhs, tile_execute, tile_load},
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

    fn allocate_lhs(#[comptime] config: Self::Config) -> Tile<L, VL, ReadWrite> {
        cmma_allocate_lhs(MatrixLayout::RowMajor, config.tile_size)
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Tile<R, VR, ReadWrite> {
        cmma_allocate_rhs(MatrixLayout::RowMajor, config.tile_size)
    }

    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Tile<R, VR, ReadWrite> {
        cmma_allocate_rhs(MatrixLayout::ColMajor, config.tile_size)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        source: &Tile<E, ES, ReadOnly>,
        dest: &mut Tile<L, VL, ReadWrite>,
    ) {
        tile_load::<E, ES, L, VL, L, R, A>(source, dest, StageIdent::Lhs);
    }

    fn load_rhs<E: Float, ES: Size>(
        source: &Tile<E, ES, ReadOnly>,
        dest: &mut Tile<R, VR, ReadWrite>,
    ) {
        tile_load::<E, ES, R, VR, L, R, A>(source, dest, StageIdent::Rhs);
    }

    fn execute(
        lhs: &Tile<L, VL, ReadWrite>,
        rhs: &Tile<R, VR, ReadWrite>,
        acc: &mut Tile<A, VA, ReadWrite>,
        #[comptime] _tile_size: TileSize,
    ) {
        tile_execute::<L, VL, R, VR, A, VA>(lhs, rhs, acc)
    }
}
