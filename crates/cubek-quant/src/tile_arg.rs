//! A quantized operand as the one tile a kernel reads.

use cubecl::tile::{
    TileArgLaunch,
    kind::Field,
    launch::{Arg, Labelled, Refusal},
};
use cubecl_common::quant::scheme::QuantScheme;

/// `values` under their `scales`, finest level first, as the one tile a kernel reads: the values
/// packed as `scheme` stores them, each scale level packed as the scheme's field for it and
/// nested under the level before. Each argument states its own axes, so a level spans the axes it
/// distinguishes and is broadcast over the ones it omits.
pub fn quantized_arg<'a>(
    scheme: &QuantScheme,
    values: Arg<'a, Labelled>,
    scales: impl IntoIterator<Item = Arg<'a, Labelled>>,
) -> Result<TileArgLaunch, Refusal> {
    let fields = scheme
        .block_scale()
        .map(|block| block.dtype)
        .into_iter()
        .chain(scheme.tensor_scale())
        .map(Field::of_scale);
    let mut tile = values.packed(scheme.value).build()?.arg();
    for (scale, field) in scales.into_iter().zip(fields) {
        tile = tile.scales(scale.packed(field).build()?.arg());
    }
    Ok(tile)
}
