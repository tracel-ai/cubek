use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

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
        // Create accumulators for rowwise max and sum
        let max = RowWise::new_min_value(config.num_rows_per_unit());
        let sum = RowWise::new_zero(config.num_rows_per_unit());

        // Create one shared memory for between score fragment and local tile and one for
        //  between local tile and softmaxed fragment, and slice both for current plane
        let total_tile_size = (config.tile_size.seq_q * config.tile_size.seq_kv) as usize;
        let smem_size = total_tile_size * config.num_planes as usize;
        let start = UNIT_POS_Y as usize * total_tile_size;
        let end = start + total_tile_size;
        let score_smem = SharedMemory::new(smem_size).slice_mut(start, end);
        let softmaxed_smem = SharedMemory::new(smem_size).slice_mut(start, end);

        // Create a local tile for softmax computations
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
    type ScoreTile = cmma::Matrix<Acc>;
    type SoftmaxedTile = cmma::Matrix<Lhs>;
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
        cmma::store(
            &mut workspace.score_smem,
            score_matmul_accumulator,
            config.tile_size.seq_kv,
            cmma::MatrixLayout::RowMajor,
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

        // Make sure the mutations on softmax_rowwise also affect other softmax formats
        workspace.local_tile.store_to(&mut workspace.softmaxed_smem);

        sync_cube();

        cmma::load(
            value_matmul_lhs,
            &workspace.softmaxed_smem.to_slice(),
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
        let mut tile = unsafe {
            cmma::Matrix::<Acc>::uninitialized(
                cmma::MatrixIdent::Accumulator,
                config.tile_size.seq_q as usize,
                config.tile_size.seq_kv as usize,
                config.tile_size.head_dim as usize,
                cmma::MatrixLayout::Undefined,
            )
        };
        Self::zero_score_tile(&mut tile);
        tile
    }

    fn zero_score_tile(score_tile: &mut Self::ScoreTile) {
        cmma::fill(score_tile, Acc::from_int(0));
    }

    fn init_softmax_tile(#[comptime] config: Self::Config) -> Self::SoftmaxedTile {
        unsafe {
            cmma::Matrix::<Lhs>::uninitialized(
                cmma::MatrixIdent::A,
                config.tile_size.seq_q as usize,
                config.tile_size.seq_kv as usize,
                config.tile_size.head_dim as usize,
                cmma::MatrixLayout::RowMajor,
            )
        }
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
