use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube};

/// A unit's window of `out`, every cell set to zero.
#[cube]
pub fn unit_zero_microkernel(out: &mut Tile) {
    out.zero();
}
