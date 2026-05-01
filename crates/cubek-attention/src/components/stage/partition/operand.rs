use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::Query,
    components::tile::matmul::{
        AttentionTileMatmul, allocate_lhs, allocate_rhs, allocate_rhs_transposed,
    },
    components::tile::{Key, Value},
    definition::AttentionPartitionSize,
};

#[derive(CubeType)]
/// Contains all seq_q·head_dim materialized tiles at once because they are reused extensively
pub struct QueryPartition<L: Numeric> {
    sequence: Sequence<Query<L>>,
}

#[cube]
impl<L: Numeric> QueryPartition<L> {
    pub fn new(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] matmul: AttentionTileMatmul,
    ) -> QueryPartition<L> {
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.head_dim {
            sequence.push(Query::<L>::new(allocate_lhs::<L>(matmul)));
        }

        QueryPartition::<L> { sequence }
    }

    pub fn get(
        &self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &Query<L> {
        &self.sequence[q * partition_head_dim + hd]
    }

    pub fn get_mut(
        &mut self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &mut Query<L> {
        self.sequence.index_mut(q * partition_head_dim + hd)
    }
}

#[derive(CubeType)]
pub struct KeyPartition<R: Numeric> {
    sequence: Sequence<Key<R>>,
}

#[cube]
impl<R: Numeric> KeyPartition<R> {
    pub fn new(#[comptime] matmul: AttentionTileMatmul) -> KeyPartition<R> {
        let mut keys = Sequence::new();
        keys.push(Key::<R>::new(allocate_rhs_transposed::<R>(matmul)));
        KeyPartition::<R> { sequence: keys }
    }

    pub fn get(&self) -> &Key<R> {
        &self.sequence[0usize]
    }

    pub fn get_mut(&mut self) -> &mut Key<R> {
        self.sequence.index_mut(0usize)
    }
}

#[derive(CubeType)]
pub struct ValuePartition<R: Numeric> {
    sequence: Sequence<Value<R>>,
}

#[cube]
impl<R: Numeric> ValuePartition<R> {
    pub fn new(#[comptime] matmul: AttentionTileMatmul) -> ValuePartition<R> {
        let mut values = Sequence::new();
        values.push(Value::<R>::new(allocate_rhs::<R>(matmul)));
        ValuePartition::<R> { sequence: values }
    }

    pub fn get(&self) -> &Value<R> {
        &self.sequence[0usize]
    }

    pub fn get_mut(&mut self) -> &mut Value<R> {
        self.sequence.index_mut(0usize)
    }
}
