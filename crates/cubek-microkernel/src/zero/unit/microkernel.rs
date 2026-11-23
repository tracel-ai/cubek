use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube};

#[cube]
pub(crate) fn unit_direct_zero_microkernel(out: &mut Tile) {
    out.zero();
}
