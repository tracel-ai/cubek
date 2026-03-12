use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::output::AttentionOutput;
use crate::components::tile::pipeline::{LocalTile, RowWise};
use crate::definition::AttentionTileSize;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BlackboxOutputConfig {
    pub tile_size: AttentionTileSize,
}

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct BlackboxAttentionOutput<SM: Float, Acc: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(SM, Acc)>,
}

#[derive(CubeType)]
pub struct AccumulatorProcedureWorkspace<Acc: Float> {
    acc_smem_slice: SliceMut<Acc>,
    local_tile: LocalTile<Acc>,
}

#[cube]
impl<SM: Float, Acc: Float> AttentionOutput for BlackboxAttentionOutput<SM, Acc> {
    type Config = BlackboxOutputConfig;
    type ScaleColumn = RowWise<SM>;
    type RunningState = (RowWise<SM>, RowWise<SM>);
    type Tile = cmma::Matrix<Acc>;
    type Workspace = AccumulatorProcedureWorkspace<Acc>;

    fn scale_mul(
        tile: &mut Self::Tile,
        scale: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        cmma::store(
            &mut workspace.acc_smem_slice,
            &tile,
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.acc_smem_slice.to_slice());

        sync_cube();

        workspace
            .local_tile
            .rowwise_scale(&RowWise::<SM>::cast_from(&scale));

        workspace.local_tile.store_to(&mut workspace.acc_smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &tile,
            &workspace.acc_smem_slice.to_slice(),
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn scale_div(
        tile: &mut Self::Tile,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let mut scale = RowWise::<SM>::cast_from(&running_state.1);
        scale.recip_inplace();

        cmma::store(
            &mut workspace.acc_smem_slice,
            &tile,
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.acc_smem_slice.to_slice());

        sync_cube();

        workspace.local_tile.rowwise_scale(&scale);

        workspace.local_tile.store_to(&mut workspace.acc_smem_slice);

        sync_cube();

        cmma::load_with_layout(
            &tile,
            &workspace.acc_smem_slice.to_slice(),
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        todo!()
    }

    fn init_tile(#[comptime] config: Self::Config) -> Self::Tile {
        todo!()
    }

    fn write_results<E: Float, ES: Size>(
        tile: &Self::Tile,
        slice: &mut SliceMut<Vector<E, ES>>,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }
}
