use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile_matmul::{
    Plane, Tile, TileMatmul, mma::config::MmaMatmulConfig, mma_allocate_acc, mma_allocate_lhs,
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
    type Scope = Plane;

    fn execute(
        lhs: &Tile<L, VL, Self::Scope, ReadWrite>,
        rhs: &Tile<R, VR, Self::Scope, ReadWrite>,
        acc: &mut Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_execute::<L, VL, R, VR, A, VA, Self::Scope>(lhs, rhs, acc);
    }

    fn allocate_lhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<L, VL, Self::Scope, ReadWrite> {
        mma_allocate_lhs::<L, VL, R, A, Self::Scope>(layout, config.shared, config.mma_io_config)
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<R, VR, Self::Scope, ReadWrite> {
        mma_allocate_rhs::<R, VR, L, A, Self::Scope>(layout, config.shared, config.mma_io_config)
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<A, VA, Self::Scope, ReadWrite> {
        mma_allocate_acc::<A, VA, L, R, Self::Scope>(layout, config.shared, config.mma_io_config)
    }

    fn load_lhs<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        lhs: &mut Tile<L, VL, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_load::<E, ES, L, VL, L, R, A, Self::Scope>(tile, lhs, StageIdent::Lhs);
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tile<R, VR, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_load::<E, ES, R, VR, L, R, A, Self::Scope>(tile, rhs, StageIdent::Rhs);
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_load::<E, ES, A, VA, L, R, A, Self::Scope>(tile, acc, StageIdent::Acc);
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tile<E, ES, Self::Scope, ReadWrite>,
        out: &mut Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_write::<E, ES, A, VA, L, R, Self::Scope>(tile, out);
    }
}
