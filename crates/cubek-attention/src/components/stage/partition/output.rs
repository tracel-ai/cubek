use cubecl;
use cubecl::prelude::*;
use cubek_std::tile::{Plane, RowWise, RowwiseTileKind, RowwiseTileWorkspace, Tile};

use crate::components::tile::matmul::{self as attn_matmul, AttentionTileMatmul};
use crate::definition::AttentionPartitionSize;

#[derive(CubeType)]
/// Holds the per-partition output accumulator tiles plus a workspace for the
/// row-wise scaling round-trip.
pub struct OutputPartition<Acc: Float, VA: Size> {
    workspace: RowwiseTileWorkspace<Acc>,
    sequence: Sequence<Tile<Acc, VA, Plane, ReadWrite>>,
}

#[cube]
impl<Acc: Float, VA: Size> OutputPartition<Acc, VA> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] value_matmul: AttentionTileMatmul,
        #[comptime] rowwise_kind: RowwiseTileKind,
    ) -> OutputPartition<Acc, VA> {
        let mut sequence = Sequence::new();
        let workspace = RowwiseTileWorkspace::<Acc>::new(rowwise_kind);

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.val_dim {
            // Output accumulator = value matmul accumulator. The matmul variant
            // chooses Register vs Cmma allocation automatically.
            let mut tile = attn_matmul::allocate_acc::<Acc, VA>(value_matmul);
            tile.fill_zero();
            sequence.push(tile);
        }

        OutputPartition::<Acc, VA> {
            workspace,
            sequence,
        }
    }

    pub fn get_at(
        &self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &Tile<Acc, VA, Plane, ReadWrite> {
        &self.sequence[i * partition_val_dim + j]
    }

    pub fn get_at_mut(
        &mut self,
        #[comptime] i: usize,
        #[comptime] j: usize,
        #[comptime] partition_val_dim: usize,
    ) -> &mut Tile<Acc, VA, Plane, ReadWrite> {
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
            .scale_mul::<SM>(scale, &mut self.workspace);
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
            .scale_div::<SM>(running_state_l, &mut self.workspace);
    }
}
