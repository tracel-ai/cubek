use crate::definition::{Resample, Semiring};
use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDyn},
};

/// Resample instruction that accumulates tap weights to produce a single tap value.
pub struct ResampleInstruction;

/// Accumulator that stores the accumulated tap weights and arguments.
#[derive(CubeType)]
#[allow(dead_code)]
pub struct Accumulator<F: Float, N: Size> {
    pub elements: Value<Vector<F, N>>,
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

    pub fn assign(&mut self, other: &Value<T>) {
        match (self, other) {
            (Value::Multiple(this), Value::Multiple(other)) => {
                for i in 0..this.len() {
                    this[i] = other[i];
                }
            }
            (Value::Single(this), Value::Single(other)) => {
                this.value = other.value;
            }
            (Value::None, Value::None) => {}
            _ => panic!("Tried assigning different accumulator kinds"),
        }
    }
}

#[cube]
impl ResampleInstruction {
    pub fn initialize<F: Float, N: Size>(#[comptime] config: &Resample) -> Accumulator<F, N> {
        let identity = Semiring::identity(&config.semiring);

        Accumulator::<F, N> {
            elements: Value::new_single(identity),
            args: Value::new_None(),
        }
    }

    pub fn combine<F: Float, N: Size>(
        value: &mut Vector<F, N>,
        weight: Vector<F, N>,
        _tap_idx: usize,
        #[comptime] config: &Resample,
    ) {
        *value = Semiring::combine(*value, weight, &config.semiring)
    }

    pub fn accumulate<F: Float, N: Size>(
        accumulator: &mut Accumulator<F, N>,
        combined: Vector<F, N>,
        _tap_idx: usize,
        #[comptime] config: &Resample,
    ) {
        let elements = accumulator.elements.item();
        let accumulated = Semiring::accumulate(elements, combined, &config.semiring);

        accumulator.elements.assign(&Value::new_single(accumulated));
    }

    pub fn count_position<F: Float, N: Size>(
        _accumulator: &mut Accumulator<F, N>,
        _position: &CoordsDyn,
        #[comptime] _config: &Resample,
    ) {
    }

    pub fn store<F: Float, N: Size>(
        out_coord: CoordsDyn,
        output: &mut ViewMut<Vector<F, N>, CoordsDyn>,
        accumulator: Accumulator<F, N>,
        #[comptime] _config: &Resample,
    ) {
        output.write(out_coord, accumulator.elements.item());
    }
}
