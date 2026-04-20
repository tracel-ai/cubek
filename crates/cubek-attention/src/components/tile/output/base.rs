use cubecl;
use cubecl::prelude::*;
<<<<<<< HEAD
use cubek_matmul::components::tile_matmul::{Plane, Tile};
=======
use cubek_matmul::components::tile::Tile;
>>>>>>> main

#[cube]
pub trait AttentionOutput<A: Float, VA: Size>: Send + Sync + 'static + Sized {
    type Config: Copy + Clone;
    type ScaleColumn: CubeType;
    type RunningState: CubeType;
    type Workspace: CubeType;

    fn scale_mul(
<<<<<<< HEAD
        tile: &mut Tile<A, VA, Plane, ReadWrite>,
=======
        tile: &mut Tile<A, VA, ReadWrite>,
>>>>>>> main
        column: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn scale_div(
<<<<<<< HEAD
        tile: &mut Tile<A, VA, Plane, ReadWrite>,
=======
        tile: &mut Tile<A, VA, ReadWrite>,
>>>>>>> main
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace;

<<<<<<< HEAD
    fn init_tile(#[comptime] config: Self::Config) -> Tile<A, VA, Plane, ReadWrite>;

    fn write_results<E: Float, ES: Size>(
        source: &mut Tile<A, VA, Plane, ReadWrite>,
        dest: &mut Tile<E, ES, Plane, ReadWrite>,
=======
    fn init_tile(#[comptime] config: Self::Config) -> Tile<A, VA, ReadWrite>;

    fn write_results<E: Float, ES: Size>(
        source: &mut Tile<A, VA, ReadWrite>,
        dest: &mut Tile<E, ES, ReadWrite>,
>>>>>>> main
        #[comptime] config: Self::Config,
    );
}
