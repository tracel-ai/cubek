use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::{Plane, RowWise, SoftmaxKind, Tile, softmax_init_state};

use crate::components::tile::matmul::{self as attn_matmul, AttentionTileMatmul};
use crate::{components::tile::MaskTile, definition::AttentionPartitionSize};

#[derive(CubeType)]
/// Holds the per-partition score and softmaxed tiles. For the cmma path each
/// tile is a `Tile::Bounce`, which encapsulates the smem + LocalTile bouncing
/// internally.
pub struct SoftmaxPartition<Acc: Float, Lhs: Float> {
    score_tiles: Sequence<Tile<Acc, Plane, ReadWrite>>,
    softmaxed_tiles: Sequence<Tile<Lhs, Plane, ReadWrite>>,
}

#[cube]
impl<Acc: Float, Lhs: Float> SoftmaxPartition<Acc, Lhs> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] score_matmul: AttentionTileMatmul,
        #[comptime] value_matmul: AttentionTileMatmul,
    ) -> SoftmaxPartition<Acc, Lhs> {
        let mut score_tiles = Sequence::new();
        let mut softmaxed_tiles = Sequence::new();

        #[unroll]
        for _ in 0..partition_size.seq_q {
            let mut score = attn_matmul::allocate_rowwise_acc::<Acc>(score_matmul);
            score.fill_zero();
            score_tiles.push(score);

            // Softmaxed tile = value matmul lhs
            softmaxed_tiles.push(attn_matmul::allocate_softmax_target_lhs::<Lhs>(
                value_matmul,
            ));
        }

        SoftmaxPartition::<Acc, Lhs> {
            score_tiles,
            softmaxed_tiles,
        }
    }

    pub fn zero_score_at(&mut self, #[comptime] q: usize) {
        self.score_tiles.index_mut(q).fill_zero();
    }

    pub fn get_score_mut(&mut self, #[comptime] q: usize) -> &mut Tile<Acc, Plane, ReadWrite> {
        self.score_tiles.index_mut(q)
    }

    pub fn get_softmaxed_mut(&mut self, #[comptime] q: usize) -> &mut Tile<Lhs, Plane, ReadWrite> {
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
            head_dim_factor,
        )
    }
}

#[cube]
pub fn init_running_state<Acc: Float>(
    #[comptime] softmax_kind: SoftmaxKind,
) -> (RowWise<Acc>, RowWise<Acc>) {
    softmax_init_state::<Acc>(softmax_kind.num_rows_per_unit())
}
