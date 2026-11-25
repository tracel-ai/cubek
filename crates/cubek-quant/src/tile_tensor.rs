//! A quantized operand as the one tile a kernel reads.

use cubecl::tile::{
    TileTensorLaunch,
    kind::Field,
    launch::{Arg, Decline, Labelled},
};
use cubecl_common::quant::scheme::QuantScheme;

/// `values` under their `scales`, finest level first, as the one tile a kernel reads: the values
/// packed as `scheme` stores them, each scale level packed as the scheme's field for it and nested
/// under the level before. Each argument states its own dimensions, so a level spans the dimensions
/// it distinguishes and is broadcast over the ones it omits.
pub fn quantized_arg<'a>(
    scheme: &QuantScheme,
    values: Arg<'a, Labelled>,
    scales: impl IntoIterator<Item = Arg<'a, Labelled>>,
) -> Result<TileTensorLaunch, Decline> {
    quantized_arg_shifted(scheme, values, scales, Vec::new())
}

/// [`quantized_arg`], every part read at `offsets`: the value of each dynamic offset its
/// projection shifts a dim by (`Offset::Dynamic`), the values' and each level's alike, as each
/// states the dims a window moves along.
pub fn quantized_arg_shifted<'a>(
    scheme: &QuantScheme,
    values: Arg<'a, Labelled>,
    scales: impl IntoIterator<Item = Arg<'a, Labelled>>,
    offsets: Vec<i32>,
) -> Result<TileTensorLaunch, Decline> {
    let fields = scheme.levels().map(|level| Field::of_scale(level.dtype));
    let shifted = |arg: TileTensorLaunch| match offsets.is_empty() {
        true => arg,
        false => arg.runtime_map(Vec::new(), offsets.clone()),
    };
    let mut tile = shifted(values.packed(scheme.value).build()?.arg());
    for (scale, field) in scales.into_iter().zip(fields) {
        tile = tile.scales(shifted(scale.packed(field).build()?.arg()));
    }
    Ok(tile)
}
