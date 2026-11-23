use cubecl::tile::{Region, Semiring, SoftwareMatmul, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// Each unit of `plane` writing `out = a · b` over its block as a memory window, whatever the cell
/// held: the leaf walks every contracted axis itself, the way a block whose operand a register
/// accumulator cannot step (a gathered one) is contracted.
#[cube]
pub fn plane_window_microkernel(
    plane: Region,
    a: &Tile,
    b: &Tile,
    out: &Tile,
    #[comptime] block: SoftwareMatmul,
) {
    for unit in plane {
        let mut accumulator = out.at(&unit).accumulating(block, Semiring::SumProd);
        accumulator.mm(&a.at(&unit), &b.at(&unit));
    }
}
