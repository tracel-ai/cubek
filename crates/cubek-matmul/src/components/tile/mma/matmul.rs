use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile::{
    Tile, TileMatmul, mma::config::MmaMatmulConfig, mma_allocate_acc, mma_allocate_lhs,
    mma_allocate_rhs, tile_execute, tile_load, tile_write,
};
use crate::definition::StageIdent;

/// Uses one plane to perform a small matmul using accelerated instructions, with manual register
/// management.
/// Currently requires matrix layout to match the platform's preferred layout.
pub struct MmaMatmul {}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for MmaMatmul
{
    type Config = MmaMatmulConfig;

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
        mma_allocate_lhs::<L, VL, R, A>(layout, config.shared, config.mma_io_config)
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<R, VR, ReadWrite> {
        mma_allocate_rhs::<R, VR, L, A>(layout, config.shared, config.mma_io_config)
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<A, VA, ReadWrite> {
        mma_allocate_acc::<A, VA, L, R>(layout, config.shared, config.mma_io_config)
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
