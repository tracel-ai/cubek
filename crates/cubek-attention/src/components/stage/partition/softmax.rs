use cubecl;
use cubecl::prelude::*;

use crate::components::softmax::Softmax;
use crate::components::softmax::SoftmaxConfig;
use crate::definition::AttentionPartitionSize;

#[derive(CubeType)]
/// Because at each hd we will perform matmul with all of seq_q, we keep seq_q softmax tiles at a time.
/// Each of the seq_kv column can be done sequentially reusing those tiles.
pub struct SoftmaxPartition<F: Float, SMX: Softmax<F>> {
    tiles: Sequence<SoftmaxTiles<F, SMX>>,
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
        let mut tiles = Sequence::new();

        let mut workspace = SMX::init_workspace(config);

        #[unroll]
        for _ in 0..partition_size.seq_q {
            tiles.push(SoftmaxTiles::<F, SMX> {
                score_tile: SMX::init_score_tile(&mut workspace, config),
                softmaxed_tile: SMX::init_softmax_tile(&mut workspace, config),
            });
        }

        SoftmaxPartition::<F, SMX> { tiles }
    }

    pub fn get_tiles_mut(&mut self, #[comptime] q: usize) -> &mut SoftmaxTiles<F, SMX> {
        self.tiles.index_mut(q)
    }
}
