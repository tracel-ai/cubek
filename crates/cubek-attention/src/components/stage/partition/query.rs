use cubecl;
use cubecl::prelude::*;

use crate::components::softmax::InnerMatmul;
use crate::components::stage::QueryTile;
use crate::definition::AttentionPartitionSize;

#[derive(CubeType)]
/// Contains all seq_q·head_dim materialized tiles at once because they are reused extensively
pub struct QueryPartition<IM: InnerMatmul> {
    sequence: Sequence<QueryTile<IM>>,
}

#[cube]
impl<IM: InnerMatmul> QueryPartition<IM> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] config: IM::Config,
    ) -> QueryPartition<IM> {
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.head_dim {
            sequence.push(QueryTile::<IM>::new(config));
        }

        QueryPartition::<IM> { sequence }
    }

    pub fn get(
        &self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &QueryTile<IM> {
        &self.sequence[q * partition_head_dim + hd]
    }

    pub fn get_mut(
        &mut self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &mut QueryTile<IM> {
        self.sequence.index_mut(q * partition_head_dim + hd)
    }
}
