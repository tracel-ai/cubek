use cubecl::prelude::*;

/// Resample instruction that accumulates tap weights to produce a single tap value.
pub struct ResampleInstruction;

/// Accumulator that stores the accumulated tap weights and arguments.
#[derive(CubeType)]
#[allow(dead_code)]
pub struct Accumulator<F: Float, N: Size> {
    pub elements: Value<Vector<F, N>>,
    pub weight_accumulator: Value<Vector<F, N>>,
    pub args: Value<Vector<u32, N>>,
}

/// Whether the accumulator has zero, one or more vectors.
#[derive(CubeType)]
#[allow(dead_code)]
pub enum Value<T: CubePrimitive> {
    Multiple(Array<T>),
    /// Wrap the item to be able to modify it as a field.
    Single(ValueWrapper<T>),
    None,
}

/// Wrap the item to be able to modify it as a field.
#[derive(CubeType)]
pub struct ValueWrapper<T: CubePrimitive> {
    value: T,
}

#[cube]
impl<T: CubePrimitive> Value<T> {
    pub fn new_single(value: T) -> Value<T> {
        Value::new_Single(ValueWrapper::<T> { value })
    }

    pub fn item(&self) -> T {
        match self {
            Value::Multiple(_) => panic!("Tried item on Multiple"),
            Value::Single(item) => item.value,
            Value::None => panic!("Tried item on None"),
        }
    }

    pub fn set_item(&mut self, new_value: T) {
        #[comptime]
        match self {
            Value::Single(item) => item.value = new_value,
            _ => panic!("Tried setting item on a non-Single variant"),
        }
    }
}
