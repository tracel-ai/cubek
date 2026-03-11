use cubecl;
use cubecl::prelude::*;

use crate::components::softmax::InnerMatmul;
use crate::components::stage::{Key, Value};

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
