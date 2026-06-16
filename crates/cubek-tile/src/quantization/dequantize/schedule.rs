use cubecl::prelude::*;

use crate::{Space, Tile, Walk, quantization::dequantize::Dequantize};

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
    let space = comptime![Space::merge(&[&input.space, &output.space])];
    for region in Walk::over(space) {
        output.dequantize_at(input, scales, &region);
    }
}
