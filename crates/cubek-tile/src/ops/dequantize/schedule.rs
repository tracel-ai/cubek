use cubecl::prelude::*;

use crate::*;

#[cube]
pub(crate) fn dequantize_direct<I: Numeric, O: Numeric>(input: &Tile<O>, output: &mut Tile<O>) {
    for region in Walk::over(input.runtime_space()) {
        output.dequantize_from_at::<I>(input, &region);
    }
}
