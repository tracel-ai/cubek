use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::StridedTile;

use crate::components::tile::MaskTile;
use crate::components::tile::pipeline::{LocalTileLayout, RowWise, UnitTile, UnitTileLayout};
use crate::components::tile::softmax::unit::UnitSoftmaxConfig;
use crate::components::tile::softmax::{Reducer, Softmax, UnitReducer};

#[derive(CubeType)]
pub struct UnitSoftmax<Lhs: Float> {
    #[cube(comptime)]
    _phantom: PhantomData<Lhs>,
}

#[derive(CubeType)]
pub struct UnitSoftmaxWorkspace<Acc: Float, Lhs: Float> {
    max: RowWise<Acc>,
    sum: RowWise<Acc>,
    #[cube(comptime)]
    _phantom: PhantomData<Lhs>,
}

#[cube]
impl<Acc: Float, Lhs: Float> UnitSoftmaxWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] config: UnitSoftmaxConfig) -> Self {
        UnitSoftmaxWorkspace::<Acc, Lhs> {
            max: RowWise::new_min_value(config.num_rows_per_unit as usize),
            sum: RowWise::new_zero(config.num_rows_per_unit as usize),
            _phantom: PhantomData,
        }
    }
}

#[cube]
impl<Acc: Float, Lhs: Float> Softmax<Acc> for UnitSoftmax<Lhs> {
    type Config = UnitSoftmaxConfig;
    type ScaleColumn = RowWise<Acc>;
    type RunningState = (RowWise<Acc>, RowWise<Acc>);
    type ScoreTile = UnitTile<Acc>;
    type SoftmaxedTile = UnitTile<Lhs>;
    type Workspace = UnitSoftmaxWorkspace<Acc, Lhs>;
    type Mask = UnitTile<Acc>;
    type ScoreLayout = UnitTileLayout;

    fn softmax(
        score_matmul_accumulator: &mut Self::ScoreTile,
        mask: &MaskTile<Acc, Self>,
        value_matmul_lhs: &mut Self::SoftmaxedTile,
        state: &mut Self::RunningState,
        workspace: &mut Self::Workspace,
        head_dim_factor: Acc,
        #[comptime] config: Self::Config,
    ) -> Self::ScaleColumn {
        score_matmul_accumulator.scale_and_mask::<MaskTile<Acc, Self>>(head_dim_factor, mask);

        UnitReducer::row_max(&mut workspace.max, &state.0, score_matmul_accumulator);

        score_matmul_accumulator.exp_diff(&workspace.max);

        UnitReducer::row_sum(&mut workspace.sum, score_matmul_accumulator);

        let exp_m_diff = state.0.exp_diff(&workspace.max);

        let new_l = exp_m_diff.mul(&state.1).add(&workspace.sum);

        RowWise::copy_from(&mut state.0, &workspace.max);
        RowWise::copy_from(&mut state.1, &new_l);

        exp_m_diff
    }

    fn init_workspace(#[comptime] config: Self::Config) -> Self::Workspace {
        Self::Workspace::new(config)
    }

    fn init_state(#[comptime] config: Self::Config) -> Self::RunningState {
        (
            RowWise::<Acc>::new_min_value(config.num_rows_per_unit as usize),
            RowWise::<Acc>::new_zero(config.num_rows_per_unit as usize),
        )
    }

    fn init_score_tile(
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) -> Self::ScoreTile {
        UnitTile::new(UnitTileLayout::new(
            config.tile_size.seq_q,
            config.tile_size.seq_kv,
        ))
    }

    fn zero_score_tile(score_tile: &mut Self::ScoreTile) {
        todo!()
    }

    fn init_softmax_tile(
        workspace: &mut Self::Workspace,
        #[comptime] config: Self::Config,
    ) -> Self::SoftmaxedTile {
        // TODO if Acc==Lhs this creates a new one uselessly
        UnitTile::new(UnitTileLayout::new(
            config.tile_size.seq_q,
            config.tile_size.seq_kv,
        ))
    }

    fn allocate_mask(#[comptime] config: Self::Config) -> Self::Mask {
        todo!()
    }
    fn load_mask<E: Numeric, ES: Size>(
        tile: &StridedTile<E, ES>,
        fragment: &mut Self::Mask,
        #[comptime] config: Self::Config,
    ) {
        todo!()
    }
    fn layout(#[comptime] config: Self::Config) -> Self::ScoreLayout {
        todo!()
    }
}
