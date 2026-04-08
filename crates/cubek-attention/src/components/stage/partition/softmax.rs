use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::MaskTile, components::tile::softmax::Softmax,
    definition::AttentionPartitionSize,
};

#[derive(CubeType)]
/// Because at each hd we will perform matmul with all of seq_q, we keep seq_q softmax tiles at a time.
/// Each of the seq_kv column can be done sequentially reusing those tiles.
pub struct SoftmaxPartition<F: Float, SMX: Softmax<F>> {
    workspace: SMX::Workspace,
    score_tiles: Sequence<SMX::ScoreTile>,
    softmaxed_tiles: Sequence<SMX::SoftmaxedTile>,
}

#[derive(CubeType)]
pub struct SoftmaxTiles<F: Float, SMX: Softmax<F>> {
    pub score_tile: SMX::ScoreTile,
    pub softmaxed_tile: SMX::SoftmaxedTile,
}

#[cube]
impl<F: Float, SMX: Softmax<F>> SoftmaxPartition<F, SMX> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] config: SMX::Config,
    ) -> SoftmaxPartition<F, SMX> {
        let mut score_tiles = Sequence::new();
        let mut softmaxed_tiles = Sequence::new();

        let workspace = SMX::init_workspace(config);

        #[unroll]
        for _ in 0..partition_size.seq_q {
            score_tiles.push(SMX::init_score_tile(config));
            softmaxed_tiles.push(SMX::init_softmax_tile(config));
        }

        SoftmaxPartition::<F, SMX> {
            workspace,
            score_tiles,
            softmaxed_tiles,
        }
    }

    pub fn zero_score_at(&mut self, #[comptime] q: usize) {
        SMX::zero_score_tile(self.get_score_mut(q));
    }

    pub fn get_score_mut(&mut self, #[comptime] q: usize) -> &mut SMX::ScoreTile {
        self.score_tiles.index_mut(q)
    }

    pub fn get_softmaxed_mut(&mut self, #[comptime] q: usize) -> &mut SMX::SoftmaxedTile {
        self.softmaxed_tiles.index_mut(q)
    }

    pub fn softmax_at(
        &mut self,
        state_q: &mut SMX::RunningState,
        mask_tile: &MaskTile<F, SMX>,
        head_dim_factor: F,
        #[comptime] q: usize,
        #[comptime] softmax_config: SMX::Config,
    ) -> SMX::ScaleColumn {
        SMX::softmax(
            self.score_tiles.index_mut(q),
            mask_tile,
            self.softmaxed_tiles.index_mut(q),
            state_q,
            &mut self.workspace,
            head_dim_factor,
            softmax_config,
        )
    }
}
