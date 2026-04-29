use std::marker::PhantomData;

use cubecl;
use cubecl::prelude::*;
use cubek_std::{
    MatrixLayout,
    tile::{
        BounceConfig, LocalTile, LocalTileLayout, Plane, RowWise, RowwiseTileKind,
        RowwiseTileWorkspace, RowwiseTileWorkspaceExpand, StridedTile, Tile, TileExpand,
        cmma_allocate_acc, cmma_allocate_lhs,
    },
};

use crate::{
    components::tile::MaskTile,
    components::tile::softmax::SoftmaxConfig as _,
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
    score_rowwise: RowwiseTileWorkspace<Acc>,
    softmaxed_rowwise: RowwiseTileWorkspace<Lhs>,
}

#[cube]
impl<Acc: Float, Lhs: Float> BlackboxSoftmaxWorkspace<Acc, Lhs> {
    pub fn new(#[comptime] config: BlackboxSoftmaxConfig) -> Self {
        let kind = comptime! {
            RowwiseTileKind::Bounce(BounceConfig {
                tile_shape: (config.tile_size.seq_q, config.tile_size.seq_kv),
                num_planes: config.num_planes,
                plane_dim: config.plane_dim,
                inner_layout: config.inner_layout,
            })
        };

        BlackboxSoftmaxWorkspace::<Acc, Lhs> {
            max: RowWise::new_min_value(config.num_rows_per_unit()),
            sum: RowWise::new_zero(config.num_rows_per_unit()),
            score_rowwise: RowwiseTileWorkspace::new(kind),
            softmaxed_rowwise: RowwiseTileWorkspace::new(kind),
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
        let stride = config.tile_size.seq_kv;

        // Bounce in: cmma -> smem -> local_tile, gives unit-addressable view of the score tile.
        score_matmul_accumulator.bounce_in(&mut workspace.score_rowwise, stride);

        score_matmul_accumulator.scale_and_mask::<MaskTile<Acc, Self>>(
            head_dim_factor,
            mask,
            &mut workspace.score_rowwise,
        );

        score_matmul_accumulator.row_max(
            &mut workspace.max,
            &state.0,
            &workspace.score_rowwise,
        );

        score_matmul_accumulator.exp_diff(&workspace.max, &mut workspace.score_rowwise);

        score_matmul_accumulator.row_sum(&mut workspace.sum, &workspace.score_rowwise);

        let exp_m_diff = state.0.exp_diff(&workspace.max);
        let new_l = exp_m_diff.mul(&state.1).add(&workspace.sum);

        RowWise::copy_from(&mut state.0, &workspace.max);
        RowWise::copy_from(&mut state.1, &new_l);

        // Cross-precision write into the value-matmul lhs cmma fragment via the
        // softmaxed_rowwise workspace's smem.
        copy_local_to_smem_cast::<Acc, Lhs>(
            &workspace.score_rowwise,
            &mut workspace.softmaxed_rowwise,
        );
        sync_cube();
        load_softmaxed_cmma_from_smem::<Lhs>(
            value_matmul_lhs,
            &mut workspace.softmaxed_rowwise,
            stride,
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
        score_tile.fill_zero();
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

/// Writes the score-side `local_tile` (Acc) into the softmaxed-side smem (Lhs),
/// performing the precision cast in the process.
#[cube]
fn copy_local_to_smem_cast<Acc: Float, Lhs: Float>(
    score: &RowwiseTileWorkspace<Acc>,
    softmaxed: &mut RowwiseTileWorkspace<Lhs>,
) {
    match (score, softmaxed) {
        (RowwiseTileWorkspace::Bounce(s), RowwiseTileWorkspace::Bounce(d)) => {
            s.local_tile.store_to(&mut d.smem);
        }
        _ => panic!("copy_local_to_smem_cast: expected both workspaces to be Bounce"),
    }
}

/// Loads the softmaxed-side smem (Lhs) into the value-matmul lhs cmma fragment.
#[cube]
fn load_softmaxed_cmma_from_smem<Lhs: Float>(
    tile: &mut Tile<Lhs, Const<0>, Plane, ReadWrite>,
    workspace: &mut RowwiseTileWorkspace<Lhs>,
    #[comptime] stride: u32,
) {
    match (tile, workspace) {
        (Tile::Cmma(t), RowwiseTileWorkspace::Bounce(bw)) => {
            cmma::load(&t.matrix, &bw.smem.to_slice(), stride);
        }
        _ => panic!("load_softmaxed_cmma_from_smem: expected Cmma tile + Bounce workspace"),
    }
}
