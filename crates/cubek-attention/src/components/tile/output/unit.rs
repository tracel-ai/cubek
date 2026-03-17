use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::output::AttentionOutput;
use crate::components::tile::pipeline::{RowWise, UnitTile, UnitTileLayout, unit_tile_to_slice};
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
        _workspace: &mut Self::Workspace,
        #[comptime] _config: Self::Config,
    ) {
        tile.rowwise_scale(&RowWise::<SM>::cast_from(scale));
    }

    fn scale_div(
        tile: &mut Self::Tile,
        running_state: &Self::RunningState,
        _workspace: &mut Self::Workspace,
        #[comptime] _config: Self::Config,
    ) {
        let mut scale = RowWise::<SM>::cast_from(&running_state.1);
        scale.recip_inplace();

        tile.rowwise_scale(&scale);
    }

    fn init_workspace(#[comptime] _config: Self::Config) -> Self::Workspace {}

    fn init_tile(#[comptime] config: Self::Config) -> Self::Tile {
        let mut tile = UnitTile::new(UnitTileLayout::new(
            config.tile_size.seq_q,
            config.tile_size.val_dim,
            false,
        ));
        tile.zero();
        tile
    }

    fn write_results<E: Float, ES: Size>(
        tile: &Self::Tile,
        slice: &mut SliceMut<Vector<E, ES>>,
        #[comptime] _config: Self::Config,
    ) {
        unit_tile_to_slice(tile, slice)
    }
}
