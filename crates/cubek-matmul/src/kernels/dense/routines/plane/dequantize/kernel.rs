use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};

/// Every plane of the launch writing its box of the packed matrix, its scales applied, into the
/// float one, its units taking the box's reads of the words in turns: the decode happens in the
/// copy, a read at a time. `O` is how the float matrix is reached: a buffer.
#[cube(launch_unchecked)]
pub fn plane_dequantize_kernel<ER: Numeric, VO: Size, O: Destination>(
    packed: &TileArg<'_>,
    out: &O::Arg<ER>,
    partitioning: Partitioning,
    #[define(ER)] _element: ElemType,
) {
    let packed = packed.tile(&partitioning);
    let c = O::tile::<ER, VO>(out, &partitioning);
    for plane in &partitioning {
        let mut c = c.at(&plane);
        c.dequantize_from(&packed.at(&plane));
    }
}
