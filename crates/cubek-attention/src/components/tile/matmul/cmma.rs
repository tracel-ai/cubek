use cubecl;
use cubecl::prelude::*;
use cubek_matmul::{
    components::tile_matmul::{
        Plane, Tile, cmma_allocate_lhs, cmma_allocate_rhs, tile_execute, tile_load,
    },
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

    fn allocate_lhs(#[comptime] config: Self::Config) -> Tile<L, VL, Plane, ReadWrite> {
        cmma_allocate_lhs::<L, VL, Plane>(MatrixLayout::RowMajor, config.tile_size)
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Tile<R, VR, Plane, ReadWrite> {
        cmma_allocate_rhs::<R, VR, Plane>(MatrixLayout::RowMajor, config.tile_size)
    }

    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Tile<R, VR, Plane, ReadWrite> {
        cmma_allocate_rhs::<R, VR, Plane>(MatrixLayout::ColMajor, config.tile_size)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        source: &Tile<E, ES, Plane, ReadOnly>,
        dest: &mut Tile<L, VL, Plane, ReadWrite>,
    ) {
        tile_load::<E, ES, L, VL, L, R, A, Plane>(source, dest, StageIdent::Lhs);
    }

    fn load_rhs<E: Float, ES: Size>(
        source: &Tile<E, ES, Plane, ReadOnly>,
        dest: &mut Tile<R, VR, Plane, ReadWrite>,
    ) {
        tile_load::<E, ES, R, VR, L, R, A, Plane>(source, dest, StageIdent::Rhs);
    }

    fn execute(
        lhs: &Tile<L, VL, Plane, ReadWrite>,
        rhs: &Tile<R, VR, Plane, ReadWrite>,
        acc: &mut Tile<A, VA, Plane, ReadWrite>,
        #[comptime] _tile_size: TileSize,
    ) {
        tile_execute::<L, VL, R, VR, A, VA, Plane>(lhs, rhs, acc)
    }
}
