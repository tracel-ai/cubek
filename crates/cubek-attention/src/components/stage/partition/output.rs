use cubecl;
use cubecl::prelude::*;
<<<<<<< HEAD
use cubek_matmul::components::tile_matmul::{Plane, Tile};
=======
use cubek_matmul::components::tile::Tile;
>>>>>>> main

use crate::{components::tile::output::AttentionOutput, definition::AttentionPartitionSize};

#[derive(CubeType)]
/// Contains all seq_q·val_dim materialized tiles at once because they're accumulators
pub struct OutputPartition<A: Float, VA: Size, AC: AttentionOutput<A, VA>> {
    workspace: AC::Workspace,
<<<<<<< HEAD
    sequence: Sequence<Tile<A, VA, Plane, ReadWrite>>,
=======
    sequence: Sequence<Tile<A, VA, ReadWrite>>,
>>>>>>> main
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
<<<<<<< HEAD
    ) -> &Tile<A, VA, Plane, ReadWrite> {
=======
    ) -> &Tile<A, VA, ReadWrite> {
>>>>>>> main
        &self.sequence[i * partition_val_dim + j]
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
<<<<<<< HEAD
    ) -> &mut Tile<A, VA, Plane, ReadWrite> {
=======
    ) -> &mut Tile<A, VA, ReadWrite> {
>>>>>>> main
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
