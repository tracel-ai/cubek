use crate::attention::D;
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// `sum`, a gradient summed over a walk along `D`, its `rows` rows scaled by the attention's
/// `scale`, drained into `gradient`: the query's and the keys' gradients both carry the score's
/// scale, which the walk leaves out of every step.
#[cube]
pub(crate) fn drain_scaled<EA: Float>(
    sum: &Tile,
    gradient: &Tile,
    scale: EA,
    #[comptime] rows: usize,
) {
    let mut scales = Array::<EA>::new(rows);
    #[unroll]
    for r in 0..rows {
        scales[r] = scale;
    }
    sum.along(D).mul(&scales);
    sum.drained_into(gradient);
}
