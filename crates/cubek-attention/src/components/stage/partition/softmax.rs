use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::{Plane, RowWise, SoftmaxKind, SoftmaxWorkspace, Tile, softmax_init_state};

use crate::components::tile::matmul::{self as attn_matmul, AttentionTileMatmul};
use crate::{components::tile::MaskTile, definition::AttentionPartitionSize};

#[derive(CubeType)]
/// Holds the per-partition score and softmaxed tiles plus the shared workspace
/// used by `Tile::softmax`.
pub struct SoftmaxPartition<Acc: Float, Lhs: Float> {
    workspace: SoftmaxWorkspace<Acc, Lhs>,
    score_tiles: Sequence<Tile<Acc, Const<0>, Plane, ReadWrite>>,
    softmaxed_tiles: Sequence<Tile<Lhs, Const<0>, Plane, ReadWrite>>,
}

#[cube]
impl<Acc: Float, Lhs: Float> SoftmaxPartition<Acc, Lhs> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] score_matmul: AttentionTileMatmul,
        #[comptime] value_matmul: AttentionTileMatmul,
        #[comptime] softmax_kind: SoftmaxKind,
    ) -> SoftmaxPartition<Acc, Lhs> {
        let mut score_tiles = Sequence::new();
        let mut softmaxed_tiles = Sequence::new();

        let workspace = SoftmaxWorkspace::<Acc, Lhs>::new(softmax_kind);

        #[unroll]
        for _ in 0..partition_size.seq_q {
            // Score tile = score matmul accumulator. Its variant decides whether
            // softmax bounces through smem or runs in registers.
            let mut score = attn_matmul::allocate_acc::<Acc, Const<0>>(score_matmul);
            score.fill_zero();
            score_tiles.push(score);

            // Softmaxed tile = value matmul lhs.
            softmaxed_tiles.push(attn_matmul::allocate_lhs::<Lhs, Const<0>>(value_matmul));
        }

        SoftmaxPartition::<Acc, Lhs> {
            workspace,
            score_tiles,
            softmaxed_tiles,
        }
    }

    pub fn zero_score_at(&mut self, #[comptime] q: usize) {
        self.score_tiles.index_mut(q).fill_zero();
    }

    pub fn get_score_mut(
        &mut self,
        #[comptime] q: usize,
    ) -> &mut Tile<Acc, Const<0>, Plane, ReadWrite> {
        self.score_tiles.index_mut(q)
    }

    pub fn get_softmaxed_mut(
        &mut self,
        #[comptime] q: usize,
    ) -> &mut Tile<Lhs, Const<0>, Plane, ReadWrite> {
        self.softmaxed_tiles.index_mut(q)
    }

    pub fn softmax_at(
        &mut self,
        state_q: &mut (RowWise<Acc>, RowWise<Acc>),
        mask: &MaskTile<Acc>,
        head_dim_factor: Acc,
        #[comptime] q: usize,
    ) -> RowWise<Acc> {
        self.score_tiles.index_mut(q).softmax::<Lhs, MaskTile<Acc>>(
            mask,
            self.softmaxed_tiles.index_mut(q),
            state_q,
            &mut self.workspace,
            head_dim_factor,
        )
    }
}

#[cube]
pub fn init_running_state<Acc: Float>(
    #[comptime] softmax_kind: SoftmaxKind,
) -> (RowWise<Acc>, RowWise<Acc>) {
    let n = comptime! {
        match softmax_kind {
            SoftmaxKind::Direct { num_rows_per_unit } => num_rows_per_unit,
            SoftmaxKind::Bounce(cfg) => match cfg.inner_layout {
                cubek_std::tile::InnerLayout::Contiguous => 1,
                cubek_std::tile::InnerLayout::SplitRows => 2,
            },
        }
    };
    softmax_init_state::<Acc>(n)
}
