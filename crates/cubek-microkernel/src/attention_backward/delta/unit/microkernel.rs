use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// One unit's queries: `delta = Σ_dv out_gradient · out`, summed in `EA`.
#[cube]
pub fn unit_delta_microkernel<EA: Float>(delta: &Tile, out_gradient: &Tile, out: &Tile) {
    // A tile's clone is a handle to the same cells: the dot product lands in `delta`.
    let mut delta = delta.clone();
    delta.dot::<EA>(out_gradient, out);
}
