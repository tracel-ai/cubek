use cubecl;
use cubecl::prelude::*;
use cubek_std::{TileSize, tile::StridedTile};

#[cube]
pub trait InnerMatmul {
    type Lhs: CubeType;
    type Rhs: CubeType;
    type Acc: CubeType;
    type Config: Copy + Clone;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs;
    fn load_lhs<E: Numeric, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Lhs);

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;
    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Self::Rhs;
    fn load_rhs<E: Float, ES: Size>(tile: &StridedTile<E, ES>, fragment: &mut Self::Rhs);

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Acc,
        #[comptime] tile_size: TileSize,
    );
}
