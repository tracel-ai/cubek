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
pub struct QueryPartition<L: Numeric, VL: Size> {
    sequence: Sequence<Query<L, VL>>,
}

#[cube]
impl<L: Numeric, VL: Size> QueryPartition<L, VL> {
    pub fn new<R: Numeric, VR: Size, A: Numeric, VA: Size, IM: InnerMatmul<L, VL, R, VR, A, VA>>(
        #[comptime] partition_size: AttentionPartitionSize,
        #[comptime] config: IM::Config,
    ) -> QueryPartition<L, VL> {
        let mut sequence = Sequence::new();

        #[unroll]
        for _ in 0..partition_size.seq_q * partition_size.head_dim {
            sequence.push(Query::<L, VL>::new(IM::allocate_lhs(config)));
        }

        QueryPartition::<L, VL> { sequence }
    }

    pub fn get(
        &self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &Query<L, VL> {
        &self.sequence[q * partition_head_dim + hd]
    }

    pub fn get_mut(
        &mut self,
        #[comptime] q: usize,
        #[comptime] hd: usize,
        #[comptime] partition_head_dim: usize,
    ) -> &mut Query<L, VL> {
        self.sequence.index_mut(q * partition_head_dim + hd)
    }
}

#[derive(CubeType)]
pub struct KeyPartition<R: Numeric, VR: Size> {
    sequence: Sequence<Key<R, VR>>,
}

#[cube]
impl<R: Numeric, VR: Size> KeyPartition<R, VR> {
    pub fn new<L: Numeric, VL: Size, A: Numeric, VA: Size, IM: InnerMatmul<L, VL, R, VR, A, VA>>(
        #[comptime] config: IM::Config,
    ) -> KeyPartition<R, VR> {
        let mut keys = Sequence::new();
        keys.push(Key::<R, VR>::new(IM::allocate_rhs_transposed(config)));
        KeyPartition::<R, VR> { sequence: keys }
    }

    pub fn get(&self) -> &Key<R, VR> {
        &self.sequence[0usize]
    }

    pub fn get_mut(&mut self) -> &mut Key<R, VR> {
        self.sequence.index_mut(0usize)
    }
}

#[derive(CubeType)]
pub struct ValuePartition<R: Numeric, VR: Size> {
    sequence: Sequence<Value<R, VR>>,
}

#[cube]
impl<R: Numeric, VR: Size> ValuePartition<R, VR> {
    pub fn new<L: Numeric, VL: Size, A: Numeric, VA: Size, IM: InnerMatmul<L, VL, R, VR, A, VA>>(
        #[comptime] config: IM::Config,
    ) -> ValuePartition<R, VR> {
        let mut values = Sequence::new();
        values.push(Value::<R, VR>::new(IM::allocate_rhs(config)));
        ValuePartition::<R, VR> { sequence: values }
    }

    pub fn get(&self) -> &Value<R, VR> {
        &self.sequence[0usize]
    }

    pub fn get_mut(&mut self) -> &mut Value<R, VR> {
        self.sequence.index_mut(0usize)
    }
}
