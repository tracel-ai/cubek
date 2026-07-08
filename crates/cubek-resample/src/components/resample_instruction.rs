use crate::definition::{Accumulator, NormalizationMode, Resample, Semiring, Value};
use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDynI},
};

/// Resample instruction that accumulates tap weights to produce a single tap value.
pub struct ResampleInstruction;

#[cube]
impl ResampleInstruction {
    pub fn initialize<F: Float, N: Size>(#[comptime] config: &Resample) -> Accumulator<F, N> {
        let identity = Value::new_single(Semiring::identity(&config.semiring));

        let weight_accumulator =
            NormalizationMode::initialize_weight_accumulator(&config.normalization);

        Accumulator::<F, N> {
            elements: identity,
            weight_accumulator,
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

        NormalizationMode::accumulate(
            &mut accumulator.weight_accumulator,
            weight,
            &config.normalization,
        );
    }

    pub fn count_position<F: Float, N: Size>(
        _accumulator: &mut Accumulator<F, N>,
        _position: &CoordsDynI,
        #[comptime] _config: &Resample,
    ) {
    }

    pub fn store<F: Float, N: Size>(
        out_coord: CoordsDynI,
        output: &mut ViewMut<Vector<F, N>, CoordsDynI>,
        accumulator: &Accumulator<F, N>,
        #[comptime] config: &Resample,
    ) {
        let elements = accumulator.elements.item();

        NormalizationMode::normalize(
            out_coord,
            output,
            elements,
            &accumulator.weight_accumulator,
            &config.normalization,
        );
    }
}
