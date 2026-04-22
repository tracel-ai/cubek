use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile_matmul::{
    Plane, Tile, TileMatmul, mma::config::MmaMatmulConfig, mma_allocate_acc, mma_allocate_lhs,
    mma_allocate_rhs,
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
        acc.mma(lhs, rhs);
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
        lhs.copy_from::<E, ES, L, R, A, ReadOnly>(tile, StageIdent::Lhs);
    }

    fn load_rhs<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        rhs: &mut Tile<R, VR, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        rhs.copy_from::<E, ES, L, R, A, ReadOnly>(tile, StageIdent::Rhs);
    }

    fn load_acc<E: Numeric, ES: Size>(
        tile: &Tile<E, ES, Self::Scope, ReadOnly>,
        acc: &mut Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        acc.copy_from::<E, ES, L, R, A, ReadOnly>(tile, StageIdent::Acc);
    }

    fn write_results<E: Numeric, ES: Size>(
        tile: &mut Tile<E, ES, Self::Scope, ReadWrite>,
        out: &Tile<A, VA, Self::Scope, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile.copy_from::<A, VA, L, R, A, ReadWrite>(out, StageIdent::Out);
    }
}
