use cubecl;
use cubecl::prelude::*;
<<<<<<< HEAD
use cubek_matmul::components::tile_matmul::{Plane, Tile};
=======
use cubek_matmul::components::tile::Tile;
>>>>>>> main
use cubek_std::TileSize;

#[cube]
pub trait InnerMatmul<L: Numeric, VL: Size, R: Numeric, VR: Size, A: Numeric, VA: Size> {
    type Config: Copy + Clone;

<<<<<<< HEAD
    fn allocate_lhs(#[comptime] config: Self::Config) -> Tile<L, VL, Plane, ReadWrite>;
    fn load_lhs<E: Numeric, ES: Size>(
        source: &Tile<E, ES, Plane, ReadOnly>,
        dest: &mut Tile<L, VL, Plane, ReadWrite>,
    );

    fn allocate_rhs(#[comptime] config: Self::Config) -> Tile<R, VR, Plane, ReadWrite>;
    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Tile<R, VR, Plane, ReadWrite>;
    fn load_rhs<E: Float, ES: Size>(
        source: &Tile<E, ES, Plane, ReadOnly>,
        dest: &mut Tile<R, VR, Plane, ReadWrite>,
    );

    fn execute(
        lhs: &Tile<L, VL, Plane, ReadWrite>,
        rhs: &Tile<R, VR, Plane, ReadWrite>,
        acc: &mut Tile<A, VA, Plane, ReadWrite>,
=======
    fn allocate_lhs(#[comptime] config: Self::Config) -> Tile<L, VL, ReadWrite>;
    fn load_lhs<E: Numeric, ES: Size>(
        source: &Tile<E, ES, ReadOnly>,
        dest: &mut Tile<L, VL, ReadWrite>,
    );

    fn allocate_rhs(#[comptime] config: Self::Config) -> Tile<R, VR, ReadWrite>;
    fn allocate_rhs_transposed(#[comptime] config: Self::Config) -> Tile<R, VR, ReadWrite>;
    fn load_rhs<E: Float, ES: Size>(
        source: &Tile<E, ES, ReadOnly>,
        dest: &mut Tile<R, VR, ReadWrite>,
    );

    fn execute(
        lhs: &Tile<L, VL, ReadWrite>,
        rhs: &Tile<R, VR, ReadWrite>,
        acc: &mut Tile<A, VA, ReadWrite>,
>>>>>>> main
        #[comptime] tile_size: TileSize,
    );
}
