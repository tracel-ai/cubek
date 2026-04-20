use cubecl;
use cubecl::prelude::*;
use cubek_matmul::{
    components::tile::{
        ProductType, SharedTileConfig, Tile, register_allocate_acc, register_allocate_lhs,
        register_allocate_rhs, tile_execute, tile_load,
    },
    definition::{StageIdent, SwizzleModes},
};

use crate::components::tile::matmul::InnerMatmul;

use cubek_std::{MatrixLayout, TileSize};

#[derive(CubeType)]
pub struct UnitMatmul {}

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitMatmulConfig {
    pub tile_size: TileSize,
}

impl UnitMatmulConfig {
    fn shared(&self) -> SharedTileConfig {
        SharedTileConfig::new(self.tile_size, 1, SwizzleModes::default())
    }
}

#[cube]
impl<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size>
    InnerMatmul<L, VL, R, VR, A, VA> for UnitMatmul
{
    type Config = UnitMatmulConfig;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Tile<L, VL, ReadWrite> {
        register_allocate_lhs(MatrixLayout::RowMajor, config.shared(), ProductType::Inner)
    }

    fn allocate_rhs(#[comptime] config: Self::Config) -> Tile<R, VR, ReadWrite> {
        register_allocate_rhs(MatrixLayout::RowMajor, config.shared(), ProductType::Inner)
    }

    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Tile<R, VR, ReadWrite> {
        register_allocate_rhs(MatrixLayout::ColMajor, config.shared(), ProductType::Inner)
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

#[cube]
pub fn unit_allocate_acc<A: Numeric, VA: Size>(
    #[comptime] config: UnitMatmulConfig,
) -> Tile<A, VA, ReadWrite> {
    register_allocate_acc(MatrixLayout::RowMajor, config.shared(), ProductType::Inner)
}
