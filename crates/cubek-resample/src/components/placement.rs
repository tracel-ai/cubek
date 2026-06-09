use cubecl::prelude::*;

use crate::definition::Placement;

/// Map output position to source coordinate.
#[cube]
pub fn placement_map<F: Float>(out_pos: usize, #[comptime] placement: &Placement) -> usize {
    match placement {
        Placement::Continuous { scale, offset } => usize::cast_from(
            (F::cast_from(out_pos) * F::cast_from(*scale) - F::cast_from(*offset)).floor(),
        )
        .max(0),
        Placement::Windowed { step, pad } => out_pos * *step - *pad,
    }
}
