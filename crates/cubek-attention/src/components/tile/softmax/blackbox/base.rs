use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile_matmul::{
    Plane, Tile, TileExpand, cmma_allocate_acc, cmma_allocate_lhs,
};
use cubek_std::{MatrixLayout, tile::StridedTile};

use crate::{
    components::tile::MaskTile,
    components::tile::pipeline::{LocalTile, LocalTileLayout, RowWise},
    components::tile::softmax::{BroadcastReducer, Reducer, SoftmaxConfig as _},
    components::tile::softmax::{Softmax, blackbox::BlackboxSoftmaxConfig},
};

#[derive(CubeType)]
pub struct BlackboxSoftmax<Lhs: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<Lhs>,
}

#[derive(CubeType)]
pub struct BlackboxSoftmaxWorkspace<Acc: Float, Lhs: Float> {
    max: RowWise<Acc>,
    sum: RowWise<Acc>,
    score_smem: SliceMut<Acc>,
    softmaxed_smem: SliceMut<Lhs>,
    local_tile: LocalTile<Acc>,
}

#[cube]
impl<Acc: Float, Lhs: Float> BlackboxSoftmaxWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] config: BlackboxSoftmaxConfig) -> Self {
        let max = RowWise::new_min_value(config.num_rows_per_unit());
        let sum = RowWise::new_zero(config.num_rows_per_unit());

        let total_tile_size = (config.tile_size.seq_q * config.tile_size.seq_kv) as usize;
        let smem_size = total_tile_size * config.num_planes as usize;
        let start = UNIT_POS_Y as usize * total_tile_size;
        let end = start + total_tile_size;
        let score_smem = SharedMemory::new(smem_size).slice_mut(start, end);
        let softmaxed_smem = SharedMemory::new(smem_size).slice_mut(start, end);

        let local_tile = LocalTile::new(LocalTileLayout::new(
            (config.tile_size.seq_q, config.tile_size.seq_kv),
            config.plane_dim,
            config.inner_layout,
        ));

        BlackboxSoftmaxWorkspace::<Acc, Lhs> {
            max,
            sum,
            score_smem,
            softmaxed_smem,
            local_tile,
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> Softmax<Acc> for BlackboxSoftmax<Lhs> {
    type Config = BlackboxSoftmaxConfig;
    type ScaleColumn = RowWise<Acc>;
    type RunningState = (RowWise<Acc>, RowWise<Acc>);
    type ScoreTile = Tile<Acc, Const<0>, Plane, ReadWrite>;
    type SoftmaxedTile = Tile<Lhs, Const<0>, Plane, ReadWrite>;
    type Workspace = BlackboxSoftmaxWorkspace<Acc, Lhs>;
    type Mask = LocalTile<Acc>;
    type ScoreLayout = LocalTileLayout;

    fn softmax(
        score_matmul_accumulator: &mut Self::ScoreTile,
        mask: &MaskTile<Acc, Self>,
        value_matmul_lhs: &mut Self::SoftmaxedTile,
        state: &mut Self::RunningState,
        workspace: &mut Self::Workspace,
        head_dim_factor: Acc,
        #[comptime] config: Self::Config,
    ) -> Self::ScaleColumn {
        store_cmma_to_score_smem::<Acc, Lhs>(
            score_matmul_accumulator,
            workspace,
            config.tile_size.seq_kv,
        );

        sync_cube();

        workspace
            .local_tile
            .load_from_slice(&workspace.score_smem.to_slice());

        sync_cube();

        workspace
            .local_tile
            .scale_and_mask::<MaskTile<Acc, Self>>(head_dim_factor, mask);

        BroadcastReducer::row_max(&mut workspace.max, &state.0, &workspace.local_tile);

        workspace.local_tile.exp_diff(&workspace.max);

        BroadcastReducer::row_sum(&mut workspace.sum, &workspace.local_tile);

        let exp_m_diff = state.0.exp_diff(&workspace.max);

        let new_l = exp_m_diff.mul(&state.1).add(&workspace.sum);

        RowWise::copy_from(&mut state.0, &workspace.max);
        RowWise::copy_from(&mut state.1, &new_l);

        workspace.local_tile.store_to(&mut workspace.softmaxed_smem);

        sync_cube();

        load_cmma_from_softmaxed_smem::<Acc, Lhs>(
            value_matmul_lhs,
            workspace,
            config.tile_size.seq_kv,
        );

        exp_m_diff
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        Self::Workspace::new(config)
    }

    fn init_state(#[comptime] config: Self::Config) -> Self::RunningState {
        (
            RowWise::<Acc>::new_min_value(config.num_rows_per_unit()),
            RowWise::<Acc>::new_zero(config.num_rows_per_unit()),
        )
    }

    fn init_score_tile(#[comptime] config: Self::Config) -> Self::ScoreTile {
        let mut tile = cmma_allocate_acc::<Acc, Const<0>, Plane>(
            MatrixLayout::RowMajor,
            config.tile_size.to_score_matmul_tile_size(),
        );
        Self::zero_score_tile(&mut tile);
        tile
    }

    fn zero_score_tile(score_tile: &mut Self::ScoreTile) {
        zero_cmma_score::<Acc>(score_tile);
    }

    fn init_softmax_tile(#[comptime] config: Self::Config) -> Self::SoftmaxedTile {
        cmma_allocate_lhs::<Lhs, Const<0>, Plane>(
            MatrixLayout::RowMajor,
            config.tile_size.to_score_matmul_tile_size(),
        )
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        LocalTile::new(<Self as Softmax<Acc>>::layout(config))
    }

    fn load_mask<E: Numeric, ES: Size>(
        tile: &StridedTile<E, ES>,
        fragment: &mut Self::Mask,
        #[comptime] _config: Self::Config,
    ) {
        fragment.load_from_strided_tile(tile);
    }

    fn layout(#[comptime] config: Self::Config) -> Self::ScoreLayout {
        LocalTileLayout::new(
            (config.tile_size.seq_q, config.tile_size.seq_kv),
            config.plane_dim,
            config.inner_layout,
        )
    }
}

#[cube]
fn store_cmma_to_score_smem<Acc: Float, Lhs: Float>(
    tile: &mut Tile<Acc, Const<0>, Plane, ReadWrite>,
    workspace: &mut BlackboxSoftmaxWorkspace<Acc, Lhs>,
    #[comptime] stride: u32,
) {
    match tile {
        Tile::Cmma(t) => cmma_store_score::<Acc, Lhs>(&t.matrix, workspace, stride),
        Tile::Register(_dummy) => panic!("BlackboxSoftmax expects Tile::Cmma"),
        _ => panic!("BlackboxSoftmax expects Tile::Cmma"),
    }
}

#[cube]
fn cmma_store_score<Acc: Float, Lhs: Float>(
    matrix: &cmma::Matrix<Acc>,
    workspace: &mut BlackboxSoftmaxWorkspace<Acc, Lhs>,
    #[comptime] stride: u32,
) {
    cmma::store(
        &mut workspace.score_smem,
        matrix,
        stride,
        cmma::MatrixLayout::RowMajor,
    );
}

#[cube]
fn load_cmma_from_softmaxed_smem<Acc: Float, Lhs: Float>(
    tile: &mut Tile<Lhs, Const<0>, Plane, ReadWrite>,
    workspace: &mut BlackboxSoftmaxWorkspace<Acc, Lhs>,
    #[comptime] stride: u32,
) {
    match tile {
        Tile::Cmma(t) => cmma_load_softmaxed::<Acc, Lhs>(&mut t.matrix, workspace, stride),
        Tile::Register(_dummy) => panic!("BlackboxSoftmax expects Tile::Cmma"),
        _ => panic!("BlackboxSoftmax expects Tile::Cmma"),
    }
}

#[cube]
fn cmma_load_softmaxed<Acc: Float, Lhs: Float>(
    matrix: &mut cmma::Matrix<Lhs>,
    workspace: &mut BlackboxSoftmaxWorkspace<Acc, Lhs>,
    #[comptime] stride: u32,
) {
    cmma::load(matrix, &workspace.softmaxed_smem.to_slice(), stride);
}

#[cube]
fn zero_cmma_score<Acc: Float>(tile: &mut Tile<Acc, Const<0>, Plane, ReadWrite>) {
    match tile {
        Tile::Cmma(t) => cmma::fill(&t.matrix, Acc::from_int(0)),
        Tile::Register(_dummy) => panic!("BlackboxSoftmax expects Tile::Cmma"),
        _ => panic!("BlackboxSoftmax expects Tile::Cmma"),
    }
}
