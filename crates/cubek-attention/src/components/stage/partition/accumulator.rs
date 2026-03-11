use cubecl;
use cubecl::prelude::*;

use crate::components::softmax::Accumulator;
use crate::components::softmax::AccumulatorProcedureConfig;

use crate::definition::AttentionPartitionSize;

#[derive(CubeType)]
/// Contains all seq_q·val_dim materialized tiles at once because they're accumulators
pub struct AccumulatorPartition<AC: Accumulator> {
    sequence: Sequence<AC::Tile>,
}

#[cube]
impl<AC: Accumulator> AccumulatorPartition<AC> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] config: AC::Config,
    ) -> AccumulatorPartition<AC> {
        let mut sequence = Sequence::new();

        let mut workspace = AC::init_workspace(config);

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.val_dim {
            sequence.push(AC::init_tile(&mut workspace, config));
        }

        AccumulatorPartition::<AC> { sequence }
    }

    pub fn get_at(
        &self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &AC::Tile {
        &self.sequence[i * partition_val_dim + j]
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &mut AC::Tile {
        self.sequence.index_mut(i * partition_val_dim + j)
    }
}
