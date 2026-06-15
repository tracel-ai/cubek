use cubecl::prelude::*;
use cubek_tile::TileArg;

#[cube(launch)]
/// input: the quantized input tensor
/// scales: the scale grid
/// output: the dequantized output tensor
pub fn dequantize<I: Numeric, S: Numeric, O: Numeric, IN: Size, SN: Size, ON: Size>(
    input: &TileArg<'_, I, IN>,
    scales: &TileArg<'_, S, SN>,
    output: &TileArg<'_, O, ON>,
    #[define(I)] _input_dtype: StorageType,
    #[define(S)] _scale_dtype: StorageType,
    #[define(O)] _output_dtype: StorageType,
    #[define(IN)] _input_size: usize,
    #[define(SN)] _scale_size: usize,
    #[define(ON)] _output_size: usize,
) {
    let input = input.tile();
    let scales = scales.tile();
    let mut output = output.tile();
    output.dequantize3(&input, &scales);
}
