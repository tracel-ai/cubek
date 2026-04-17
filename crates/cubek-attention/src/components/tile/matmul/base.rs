use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile::Tilex;
use cubek_std::TileSize;

#[cube]
pub trait InnerMatmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> {
    type Config: Copy + Clone;

    fn allocate_lhs(#[comptime] config: Self::Config) -> Tilex<L, VL, ReadWrite>;
    fn load_lhs<E: Numeric, ES: Size>(
        source: &Tilex<E, ES, ReadOnly>,
        dest: &mut Tilex<L, VL, ReadWrite>,
    );

    fn allocate_rhs(#[comptime] config: Self::Config) -> Tilex<R, VR, ReadWrite>;
    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Tilex<R, VR, ReadWrite>;
    fn load_rhs<E: Float, ES: Size>(
        source: &Tilex<E, ES, ReadOnly>,
        dest: &mut Tilex<R, VR, ReadWrite>,
    );

    fn execute(
        lhs: &Tilex<L, VL, ReadWrite>,
        rhs: &Tilex<R, VR, ReadWrite>,
        acc: &mut Tilex<A, VA, ReadWrite>,
        #[comptime] tile_size: TileSize,
    );
}
