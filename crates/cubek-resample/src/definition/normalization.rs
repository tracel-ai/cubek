use crate::definition::Value;
use cubecl::{
    prelude::*,
    std::tensor::{ViewMut, layout::CoordsDynI},
};

/// Normalization mode for tap weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum NormalizationMode {
    /// Preserve the kernel weights exactly.
    None,
    /// Divide by the accumulated valid weight.
    Renormalize,
}

#[cube]
impl NormalizationMode {
    pub fn initialize_weight_accumulator<F: Float, N: Size>(
        #[comptime] this: &NormalizationMode,
    ) -> Value<Vector<F, N>> {
        match this {
            NormalizationMode::None => Value::new_None(),
            NormalizationMode::Renormalize => Value::new_single(Vector::zeroed()),
        }
    }

    pub fn accumulate<F: Float, N: Size>(
        weight_accumulator: &mut Value<Vector<F, N>>,
        weight: Vector<F, N>,
        #[comptime] this: &NormalizationMode,
    ) {
        if *this == NormalizationMode::Renormalize {
            let weight = weight_accumulator.item() + weight;
            weight_accumulator.set_item(weight);
        }
    }

    pub fn normalize<F: Float, N: Size>(
        out_coord: CoordsDynI,
        output: &mut ViewMut<Vector<F, N>, CoordsDynI>,
        elements: Vector<F, N>,
        weight_accumulator: &Value<Vector<F, N>>,
        #[comptime] this: &NormalizationMode,
    ) {
        match this {
            NormalizationMode::None => output.write(out_coord, elements),
            NormalizationMode::Renormalize => {
                output.write(out_coord, elements / weight_accumulator.item())
            }
        }
    }
}
