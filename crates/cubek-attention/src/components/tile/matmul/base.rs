use cubecl;
use cubecl::prelude::*;
use cubek_std::TileSize;
use cubek_std::tile::StridedTile;

#[cube]
pub trait InnerMatmul {
    type Lhs: CubeType;
    type Rhs: CubeType;
    type Acc: CubeType;
    type Config: Copy + Clone;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Self::Lhs;
    fn load_lhs<E: Numeric>(tile: &StridedTile<E>, fragment: &mut Self::Lhs);

    fn allocate_rhs(#[comptime] config: Self::Config) -> Self::Rhs;
    fn load_rhs_plain<E: Float>(tile: &StridedTile<E>, fragment: &mut Self::Rhs);
    fn load_rhs_transposed<E: Float>(tile: &StridedTile<E>, fragment: &mut Self::Rhs);

    fn execute(
        lhs: &Self::Lhs,
        rhs: &Self::Rhs,
        out: &mut Self::Acc,
        #[comptime] tile_size: TileSize,
    );
}
