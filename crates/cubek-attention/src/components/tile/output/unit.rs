use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::output::AttentionOutput;
use crate::components::tile::pipeline::{RowWise, UnitTile};
use crate::definition::AttentionTileSize;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct UnitOutputConfig {
    pub tile_size: AttentionTileSize,
}

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct UnitAttentionOutput<SM: Float, Acc: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(SM, Acc)>,
}

#[cube]
impl<SM: Float, Acc: Float> AttentionOutput for UnitAttentionOutput<SM, Acc> {
    type Config = UnitOutputConfig;
    type ScaleColumn = RowWise<SM>;
    type RunningState = (RowWise<SM>, RowWise<SM>);
    type Tile = UnitTile<Acc>;
    type Workspace = ();

    fn scale_mul(
        tile: &mut Self::Tile,
        scale: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        tile.rowwise_scale(&RowWise::<SM>::cast_from(&scale));
    }

    fn scale_div(
        tile: &mut Self::Tile,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let mut scale = RowWise::<SM>::cast_from(&running_state.1);
        scale.recip_inplace();

        tile.rowwise_scale(&scale);
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {}

    fn init_tile(#[comptime] config: Self::Config) -> Self::Tile {
        todo!()
    }

    fn write_results<E: Float>(
        tile: &Self::Tile,
        slice: &mut SliceMut<Line<E>>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }
}
