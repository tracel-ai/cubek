use cubecl;
use cubecl::prelude::*;

use crate::components::tile::TileAttention;

use crate::components::stage::StageAttentionConfig;
use crate::components::stage::{AccumulatorTile, PartitionAttentionConfig};
use crate::definition::AttentionPrecision;

#[derive(CubeType)]
/// Contains all seq_q·val_dim materialized tiles at once because they're accumulators
pub struct AccumulatorPartition<AP: AttentionPrecision, TA: TileAttention<AP>> {
    sequence: Sequence<AccumulatorTile<AP, TA>>,
}

#[cube]
impl<AP: AttentionPrecision, TA: TileAttention<AP>> AccumulatorPartition<AP, TA> {
    pub fn new(
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> AccumulatorPartition<AP, TA> {
        let p = config.shared().partition_size;
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..p.seq_q * p.val_dim {
            sequence.push(AccumulatorTile::new(config.tile_config()));
        }

        AccumulatorPartition::<AP, TA> { sequence }
    }

    pub fn get_at(
        &self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> &AccumulatorTile<AP, TA> {
        let partition_val_dim = config.shared().partition_size.val_dim as usize;
        &self.sequence[i * partition_val_dim + j]
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] config: PartitionAttentionConfig<TA::Config>,
    ) -> &mut AccumulatorTile<AP, TA> {
        let partition_val_dim = config.shared().partition_size.val_dim as usize;
        self.sequence.index_mut(i * partition_val_dim + j)
    }
}
