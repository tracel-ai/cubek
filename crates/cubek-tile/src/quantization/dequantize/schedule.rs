use cubecl::prelude::*;

use crate::{dequantize::Dequantize, *};

#[cube]
pub(crate) fn dequantize_direct<I: Numeric, S: Numeric, O: Numeric + Dequantize<I, S>>(
    input: &Tile<I>,
    scales: &Tile<S>,
    output: &mut Tile<O>,
) {
    for region in Walk::over(input.runtime_space()) {
        output.dequantize_at(input, scales, &region);
    }
}
