use cubecl::prelude::*;

use crate::definition::Placement;

/// Map output position to source coordinate.
#[cube]
pub fn placement_map<C: Float>(out_pos: usize, #[comptime] placement: Placement) -> usize {
    match placement {
        Placement::Continuous { scale, offset } => usize::cast_from(
            (C::cast_from(out_pos) * C::cast_from(scale) - C::cast_from(offset)).floor(),
        )
        .max(0),
        Placement::Windowed { step, pad } => out_pos * step - pad,
    }
}
