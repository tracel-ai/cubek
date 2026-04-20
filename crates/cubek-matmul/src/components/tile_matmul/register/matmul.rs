use cubecl::prelude::*;
use cubek_std::MatrixLayout;

use crate::components::tile_matmul::{
    Tile, TileMatmul, Unit, register::config::RegisterMatmulConfig, register_allocate_acc,
    register_allocate_lhs, register_allocate_rhs, tile_execute, tile_load, tile_write,
};
use crate::definition::StageIdent;

/// Performs a small matmul using registers.
pub struct RegisterMatmul {}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    TileMatmul<L, VL, R, VR, A, VA> for RegisterMatmul
{
    type Config = RegisterMatmulConfig;
    type Scope = Unit;

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
        register_allocate_lhs::<L, VL, Self::Scope>(layout, config.shared, config.product_type)
    }

    fn allocate_rhs(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<R, VR, Self::Scope, ReadWrite> {
        register_allocate_rhs::<R, VR, Self::Scope>(layout, config.shared, config.product_type)
    }

    fn allocate_acc(
        #[comptime] layout: MatrixLayout,
        #[comptime] config: Self::Config,
    ) -> Tile<A, VA, Self::Scope, ReadWrite> {
        register_allocate_acc::<A, VA, Self::Scope>(layout, config.shared, config.product_type)
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
