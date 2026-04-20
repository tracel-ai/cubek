use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile::Tile;

#[cube]
pub trait AttentionOutput<A: Float, VA: Size>: Send + Sync + 'static + Sized {
    type Config: Copy + Clone;
    type ScaleColumn: CubeType;
    type RunningState: CubeType;
    type Workspace: CubeType;

    fn scale_mul(
        tile: &mut Tile<A, VA, ReadWrite>,
        column: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn scale_div(
        tile: &mut Tile<A, VA, ReadWrite>,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace;

    fn init_tile(#[comptime] config: Self::Config) -> Tile<A, VA, ReadWrite>;

    fn write_results<E: Float, ES: Size>(
        source: &mut Tile<A, VA, ReadWrite>,
        dest: &mut Tile<E, ES, ReadWrite>,
        #[comptime] config: Self::Config,
    );
}
