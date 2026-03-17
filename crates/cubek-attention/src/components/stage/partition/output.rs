use cubecl;
use cubecl::prelude::*;

use crate::components::tile::output::AttentionOutput;
use crate::definition::AttentionPartitionSize;

#[derive(CubeType)]
/// Contains all seq_q·val_dim materialized tiles at once because they're accumulators
pub struct OutputPartition<AC: AttentionOutput> {
    workspace: AC::Workspace,
    sequence: Sequence<AC::Tile>,
}

#[cube]
impl<AC: AttentionOutput> OutputPartition<AC> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] config: AC::Config,
    ) -> OutputPartition<AC> {
        let mut sequence = Sequence::new();

        let workspace = AC::init_workspace(config);

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.val_dim {
            sequence.push(AC::init_tile(config));
        }

        OutputPartition::<AC> {
            workspace,
            sequence,
        }
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

    pub fn scale_mul_at(
        &mut self,
        scale: &AC::ScaleColumn,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
        #[comptime] config: AC::Config,
    ) {
        AC::scale_mul(
            self.sequence.index_mut(i * partition_val_dim + j),
            scale,
            &mut self.workspace,
            config,
        );
    }

    pub fn scale_div_at(
        &mut self,
        running_state: &AC::RunningState,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
        #[comptime] config: AC::Config,
    ) {
        AC::scale_div(
            self.sequence.index_mut(i * partition_val_dim + j),
            running_state,
            &mut self.workspace,
            config,
        );
    }
}
