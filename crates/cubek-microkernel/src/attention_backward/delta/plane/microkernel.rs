use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's queries, one a unit: `delta = Σ_dv out_gradient · out`, summed in `EA`. Each tile
/// is windowed to `plane`.
#[cube]
pub fn plane_delta_microkernel<EA: Float>(
    plane: &Region,
    delta: &Tile,
    out_gradient: &Tile,
    out: &Tile,
) {
    for unit in plane {
        // A tile's clone is a handle to the same cells: the dot product lands in `delta`.
        let mut delta = delta.at(&unit);
        delta.dot::<EA>(&out_gradient.at(&unit), &out.at(&unit));
    }
}
