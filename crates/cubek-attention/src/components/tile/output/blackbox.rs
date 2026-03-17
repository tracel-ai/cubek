use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;

use crate::components::tile::output::AttentionOutput;
use crate::components::tile::pipeline::{InnerLayout, LocalTile, LocalTileLayout, RowWise};
use crate::definition::AttentionTileSize;

#[derive(Copy, Clone, Debug, Hash, PartialEq, Eq)]
pub struct BlackboxOutputConfig {
    pub tile_size: AttentionTileSize,
    pub num_planes: u32,
    pub plane_dim: u32,
    pub inner_layout: InnerLayout,
}

#[derive(CubeType)]
/// Accumulator tile for Tile Attention
pub struct BlackboxAttentionOutput<SM: Float, Acc: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<(SM, Acc)>,
}

#[derive(CubeType)]
pub struct BlackboxAttentionOutputWorkspace<Acc: Float> {
    smem: SliceMut<Acc>,
    local_tile: LocalTile<Acc>,
}

#[cube]
impl<Acc: Float> BlackboxAttentionOutputWorkspace<Acc> {
    fn new(#[comptime] config: BlackboxOutputConfig) -> BlackboxAttentionOutputWorkspace<Acc> {
        // Create a shared memory for going back and forth to local tile and slice for current plane
        let total_tile_size = (config.tile_size.seq_q * config.tile_size.val_dim) as usize;
        let smem_size = total_tile_size * config.num_planes as usize;
        let start = UNIT_POS_Y as usize * total_tile_size;
        let end = start + total_tile_size;
        let smem = SharedMemory::new(smem_size).slice_mut(start, end);

        let local_tile = LocalTile::new(LocalTileLayout::new(
            (config.tile_size.seq_q, config.tile_size.val_dim),
            config.plane_dim,
            config.inner_layout,
        ));

        BlackboxAttentionOutputWorkspace::<Acc> { smem, local_tile }
    }
}

#[cube]
impl<SM: Float, Acc: Float> AttentionOutput for BlackboxAttentionOutput<SM, Acc> {
    type Config = BlackboxOutputConfig;
    type ScaleColumn = RowWise<SM>;
    type RunningState = (RowWise<SM>, RowWise<SM>);
    type Tile = cmma::Matrix<Acc>;
    type Workspace = BlackboxAttentionOutputWorkspace<Acc>;

    fn scale_mul(
        tile: &mut Self::Tile,
        scale: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        cmma::store(
            &mut workspace.smem,
            tile,
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.smem.to_slice());

        sync_cube();

        workspace
            .local_tile
            .rowwise_scale(&RowWise::<SM>::cast_from(scale));

        workspace.local_tile.store_to(&mut workspace.smem);

        sync_cube();

        cmma::load_with_layout(
            tile,
            &workspace.smem.to_slice(),
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
            &mut workspace.smem,
            tile,
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.smem.to_slice());

        sync_cube();

        workspace.local_tile.rowwise_scale(&scale);

        workspace.local_tile.store_to(&mut workspace.smem);

        sync_cube();

        cmma::load_with_layout(
            tile,
            &workspace.smem.to_slice(),
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        )
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        Self::Workspace::new(config)
    }

    fn init_tile(#[comptime] config: Self::Config) -> Self::Tile {
        let tile = unsafe {
            cmma::Matrix::<Acc>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                config.tile_size.seq_q as usize,
                config.tile_size.val_dim as usize,
                config.tile_size.seq_kv as usize,
                cmma::MatrixLayout::Undefined,
            )
        };
        cmma::fill(&tile, Acc::from_int(0));
        tile
    }

    fn write_results<E: Float, ES: Size>(
        tile: &Self::Tile,
        slice: &mut SliceMut<Vector<E, ES>>,
        #[comptime] config: Self::Config,
    ) {
        let acc = cmma::cast::<Acc, E>(tile);
        cmma::store(
            slice,
            &acc,
            config.tile_size.val_dim,
            cmma::MatrixLayout::RowMajor,
        );
    }
}
