use cubecl::prelude::*;

use crate::{dequantize::Dequantize, *};

#[cube]
pub(crate) fn dequantize_direct<
    I: CubePrimitive,
    S: CubePrimitive,
    O: CubePrimitive + Dequantize<I, S>,
>(
    input: &Tile<I>,
    scales: &Tile<S>,
    output: &mut Tile<O>,
) {
    for region in Walk::over(input.runtime_space()) {
        output.dequantize_at(input, scales, &region);
    }
}
