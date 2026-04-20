use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile::{
    SharedTileConfig, Tile, TileMatmul, cmma_allocate_acc, cmma_allocate_lhs, cmma_allocate_rhs,
    tile_execute, tile_load, tile_write,
};
use crate::definition::StageIdent;

/// Uses one plane to perform a small matmul using accelerated instructions.
pub struct CmmaMatmul {}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for CmmaMatmul
{
    type Config = SharedTileConfig;

    fn execute(
        lhs: &Tile<L, VL, ReadWrite>,
        rhs: &Tile<R, VR, ReadWrite>,
        acc: &mut Tile<A, VA, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_execute::<L, VL, R, VR, A, VA>(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<L, VL, ReadWrite> {
        cmma_allocate_lhs::<L, VL>(layout, config.tile_size)
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<R, VR, ReadWrite> {
        cmma_allocate_rhs::<R, VR>(layout, config.tile_size)
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<A, VA, ReadWrite> {
        cmma_allocate_acc::<A, VA>(layout, config.tile_size)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, ReadOnly>,
        lhs: &mut Tile<L, VL, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_load::<E, ES, L, VL, L, R, A>(tile, lhs, StageIdent::Lhs);
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, ReadOnly>,
        rhs: &mut Tile<R, VR, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_load::<E, ES, R, VR, L, R, A>(tile, rhs, StageIdent::Rhs);
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, ReadOnly>,
        acc: &mut Tile<A, VA, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_load::<E, ES, A, VA, L, R, A>(tile, acc, StageIdent::Acc);
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tile<E, ES, ReadWrite>,
        out: &mut Tile<A, VA, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_write::<E, ES, A, VA, L, R>(tile, out);
    }
}
