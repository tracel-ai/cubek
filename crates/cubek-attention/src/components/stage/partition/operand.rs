use cubecl;
use cubecl::prelude::*;

use crate::{
    components::tile::Query,
    components::tile::matmul::InnerMatmul,
    components::tile::{Key, Value},
    definition::AttentionPartitionSize,
};

#[derive(CubeType)]
/// Contains all seq_q·head_dim materialized tiles at once because they are reused extensively
pub struct QueryPartition<IM: InnerMatmul> {
    sequence: Sequence<Query<IM>>,
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
            sequence.push(Query::<IM>::new(config));
        }

        QueryPartition::<IM> { sequence }
    }

    pub fn get(
        &self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &Query<IM> {
        &self.sequence[q * partition_head_dim + hd]
    }

    pub fn get_mut(
        &mut self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &mut Query<IM> {
        self.sequence.index_mut(q * partition_head_dim + hd)
    }
}

#[derive(CubeType)]
pub struct KeyPartition<IM: InnerMatmul> {
    sequence: Sequence<Key<IM>>,
}

#[cube]
impl<IM: InnerMatmul> KeyPartition<IM> {
    pub fn new(#[comptime] config: IM::Config) -> KeyPartition<IM> {
        let mut keys = Sequence::new();
        keys.push(Key::new(config));
        KeyPartition::<IM> { sequence: keys }
    }

    pub fn get(&self) -> &Key<IM> {
        &self.sequence[0usize]
    }

    pub fn get_mut(&mut self) -> &mut Key<IM> {
        self.sequence.index_mut(0usize)
    }
}

#[derive(CubeType)]
pub struct ValuePartition<IM: InnerMatmul> {
    sequence: Sequence<Value<IM>>,
}

#[cube]
impl<IM: InnerMatmul> ValuePartition<IM> {
    pub fn new(#[comptime] config: IM::Config) -> ValuePartition<IM> {
        let mut values = Sequence::new();
        values.push(Value::new(config));
        ValuePartition::<IM> { sequence: values }
    }

    pub fn get(&self) -> &Value<IM> {
        &self.sequence[0usize]
    }

    pub fn get_mut(&mut self) -> &mut Value<IM> {
        self.sequence.index_mut(0usize)
    }
}
