use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile::{Tile, TileExpand, cmma_allocate_acc, tile_write};
use cubek_std::MatrixLayout;

use crate::{
    components::tile::output::AttentionOutput,
    components::tile::pipeline::{InnerLayout, LocalTile, LocalTileLayout, RowWise},
    definition::AttentionTileSize,
};

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
impl<SM: Float, Acc: Float, VA: Size> AttentionOutput<Acc, VA>
    for BlackboxAttentionOutput<SM, Acc>
{
    type Config = BlackboxOutputConfig;
    type ScaleColumn = RowWise<SM>;
    type RunningState = (RowWise<SM>, RowWise<SM>);
    type Workspace = BlackboxAttentionOutputWorkspace<Acc>;

    fn scale_mul(
        tile: &mut Tile<Acc, VA, ReadWrite>,
        scale: &Self::ScaleColumn,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let scale_acc = RowWise::<SM>::cast_from::<Acc>(scale);
        scale_cmma_tile::<Acc, VA>(tile, &scale_acc, workspace, config);
    }

    fn scale_div(
        tile: &mut Tile<Acc, VA, ReadWrite>,
        running_state: &Self::RunningState,
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) {
        let mut scale = RowWise::<SM>::cast_from::<Acc>(&running_state.1);
        scale.recip_inplace();
        scale_cmma_tile::<Acc, VA>(tile, &scale, workspace, config);
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        Self::Workspace::new(config)
    }

    fn init_tile(#[comptime] config: Self::Config) -> Tile<Acc, VA, ReadWrite> {
        let mut tile = cmma_allocate_acc::<Acc, VA>(
            MatrixLayout::RowMajor,
            config.tile_size.to_value_matmul_tile_size(),
        );
        zero_cmma_tile::<Acc, VA>(&mut tile);
        tile
    }

    fn write_results<E: Float, ES: Size>(
        source: &mut Tile<Acc, VA, ReadWrite>,
        dest: &mut Tile<E, ES, ReadWrite>,
        #[comptime] _config: Self::Config,
    ) {
        tile_write::<E, ES, Acc, VA, Acc, Acc>(dest, source);
    }
}

#[cube]
fn scale_cmma_tile<Acc: Float, VA: Size>(
    tile: &mut Tile<Acc, VA, ReadWrite>,
    scale: &RowWise<Acc>,
    workspace: &mut BlackboxAttentionOutputWorkspace<Acc>,
    #[comptime] config: BlackboxOutputConfig,
) {
    match tile {
        Tile::Cmma(t) => scale_cmma_matrix::<Acc>(&mut t.matrix, scale, workspace, config),
        Tile::Register(_dummy) => panic!("BlackboxAttentionOutput expects a Tile::Cmma"),
        _ => panic!("BlackboxAttentionOutput expects a Tile::Cmma"),
    }
}

#[cube]
fn scale_cmma_matrix<Acc: Float>(
    matrix: &mut cmma::Matrix<Acc>,
    scale: &RowWise<Acc>,
    workspace: &mut BlackboxAttentionOutputWorkspace<Acc>,
    #[comptime] config: BlackboxOutputConfig,
) {
    cmma::store(
        &mut workspace.smem,
        matrix,
        config.tile_size.val_dim,
        cmma::MatrixLayout::RowMajor,
    );

    sync_cube();

    workspace
        .local_tile
        .load_from_slice(&workspace.smem.to_slice());

    sync_cube();

    workspace.local_tile.rowwise_scale(scale);

    workspace.local_tile.store_to(&mut workspace.smem);

    sync_cube();

    cmma::load_with_layout(
        matrix,
        &workspace.smem.to_slice(),
        config.tile_size.val_dim,
        cmma::MatrixLayout::RowMajor,
    )
}

#[cube]
fn zero_cmma_tile<Acc: Float, VA: Size>(tile: &mut Tile<Acc, VA, ReadWrite>) {
    match tile {
        Tile::Cmma(t) => cmma::fill(&t.matrix, Acc::from_int(0)),
        Tile::Register(_dummy) => panic!("BlackboxAttentionOutput expects a Tile::Cmma"),
        _ => panic!("BlackboxAttentionOutput expects a Tile::Cmma"),
    }
}
