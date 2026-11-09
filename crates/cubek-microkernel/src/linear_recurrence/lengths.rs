use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// The positions the row of a `lengths` tile has walked already: the one `u32` its window holds.
#[cube]
pub(crate) fn walked(lengths: &Tile) -> u32 {
    lengths.flat::<u32, Const<1>>().read(0usize).extract(0usize)
}
