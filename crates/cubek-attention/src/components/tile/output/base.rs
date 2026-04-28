use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::{Plane, Tile};

#[cube]
pub trait AttentionOutput<A: Float, VA: Size>: Send + Sync + 'static + Sized {
    type Config: Copy + Clone;
    type ScaleColumn: CubeType;
    type RunningState: CubeType;
    type Workspace: CubeType;

    fn scale_mul(
        tile: &mut Tile<A, VA, Plane, ReadWrite>,
        column: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn scale_div(
        tile: &mut Tile<A, VA, Plane, ReadWrite>,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    );

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace;

    fn init_tile(#[comptime] config: Self::Config) -> Tile<A, VA, Plane, ReadWrite>;

    fn write_results<E: Float, ES: Size>(
        source: &Tile<A, VA, Plane, ReadWrite>,
        dest: &mut Tile<E, ES, Plane, ReadWrite>,
        #[comptime] config: Self::Config,
    );
}
