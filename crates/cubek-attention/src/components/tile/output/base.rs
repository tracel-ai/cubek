use cubecl;
use cubecl::prelude::*;

#[cube]
pub trait AttentionOutput: Send + Sync + 'static + Sized {
    type Config: Copy + Clone;
    type ScaleColumn: CubeType;
    type RunningState: CubeType;
    type Tile: CubeType;
    type Workspace: CubeType;

    fn scale_mul(
        tile: &mut Self::Tile,
        column: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn scale_div(
        tile: &mut Self::Tile,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace;

    fn init_tile(#[comptime] config: Self::Config) -> Self::Tile;

    fn write_results<E: Float, ES: Size>(
        tile: &Self::Tile,
        slice: &mut SliceMut<Vector<E, ES>>,
        #[comptime] config: Self::Config,
    );
}
