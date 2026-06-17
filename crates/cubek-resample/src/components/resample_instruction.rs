use crate::definition::{NormalizationMode, Resample, Semiring};
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
    pub weight: Value<Vector<F, N>>,
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

#[cube]
impl ResampleInstruction {
    pub fn initialize<F: Float, N: Size>(#[comptime] config: &Resample) -> Accumulator<F, N> {
        let identity = Value::new_single(Semiring::identity(&config.semiring));

        let weight = match config.normalization {
            NormalizationMode::None => Value::new_None(),
            NormalizationMode::Renormalize => Value::new_single(Vector::zeroed()),
        };

        Accumulator::<F, N> {
            elements: identity,
            weight,
            args: Value::new_None(),
        }
    }

    pub fn combine<F: Float, N: Size>(
        value: &mut Vector<F, N>,
        weight: Vector<F, N>,
        _tap_idx: usize,
        #[comptime] config: &Resample,
    ) {
        *value = Semiring::combine(*value, weight, &config.semiring);
    }

    pub fn accumulate<F: Float, N: Size>(
        accumulator: &mut Accumulator<F, N>,
        combined: Vector<F, N>,
        weight: Vector<F, N>,
        _tap_idx: usize,
        #[comptime] config: &Resample,
    ) {
        let elements = accumulator.elements.item();
        let accumulated = Semiring::accumulate(elements, combined, &config.semiring);

        accumulator.elements.set_item(accumulated);

        if config.normalization == NormalizationMode::Renormalize {
            let weight = accumulator.weight.item() + weight;
            accumulator.weight.set_item(weight);
        }
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
        #[comptime] config: &Resample,
    ) {
        match config.normalization {
            NormalizationMode::None => output.write(out_coord, accumulator.elements.item()),
            NormalizationMode::Renormalize => output.write(
                out_coord,
                accumulator.elements.item() / accumulator.weight.item(),
            ),
        }
    }
}
