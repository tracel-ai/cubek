use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::{Plane, RowWise, Tile};

use crate::components::tile::matmul::{self as attn_matmul, AttentionTileMatmul};
use crate::definition::AttentionPartitionSize;

#[derive(CubeType)]
/// Holds the per-partition output accumulator tiles. For the cmma path each
/// tile is a `Tile::Bounce`, which carries its own smem + LocalTile internally.
pub struct OutputPartition<Acc: Float> {
    sequence: Sequence<Tile<Acc, Plane, ReadWrite>>,
}

#[cube]
impl<Acc: Float> OutputPartition<Acc> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] value_matmul: AttentionTileMatmul,
    ) -> OutputPartition<Acc> {
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.val_dim {
            let mut tile = attn_matmul::allocate_rowwise_acc::<Acc>(value_matmul);
            tile.fill_zero();
            sequence.push(tile);
        }

        OutputPartition::<Acc> { sequence }
    }

    pub fn get_at(
        &self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &Tile<Acc, Plane, ReadWrite> {
        &self.sequence[i * partition_val_dim + j]
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &mut Tile<Acc, Plane, ReadWrite> {
        self.sequence.index_mut(i * partition_val_dim + j)
    }

    pub fn scale_mul_at<SM: Float>(
        &mut self,
        scale: &RowWise<SM>,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) {
        self.sequence
            .index_mut(i * partition_val_dim + j)
            .scale_mul::<SM>(scale);
    }

    pub fn scale_div_at<SM: Float>(
        &mut self,
        running_state_l: &RowWise<SM>,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) {
        self.sequence
            .index_mut(i * partition_val_dim + j)
            .scale_div::<SM>(running_state_l);
    }
}
