use cubecl;
use cubecl::prelude::*;
use cubek_matmul::components::tile::Tilex;

use crate::{components::tile::output::AttentionOutput, definition::AttentionPartitionSize};

#[derive(CubeType)]
/// Contains all seq_q·val_dim materialized tiles at once because they're accumulators
pub struct OutputPartition<A: Float, VA: Size, AC: AttentionOutput<A, VA>> {
    workspace: AC::Workspace,
    sequence: Sequence<Tilex<A, VA, ReadWrite>>,
}

#[cube]
impl<A: Float, VA: Size, AC: AttentionOutput<A, VA>> OutputPartition<A, VA, AC> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] config: AC::Config,
    ) -> OutputPartition<A, VA, AC> {
        let mut sequence = Sequence::new();

        let workspace = AC::init_workspace(config);

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.val_dim {
            sequence.push(AC::init_tile(config));
        }

        OutputPartition::<A, VA, AC> {
            workspace,
            sequence,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &Tilex<A, VA, ReadWrite> {
        &self.sequence[i * partition_val_dim + j]
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &mut Tilex<A, VA, ReadWrite> {
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
