use cubecl::tile::{Region, Semiring, SoftwareMatmul, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// Each unit of `plane` accumulating `out += a · b` over its block as a memory window, which the
/// leaf walks along every contracted axis itself: the way a block whose operand a register
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
        let mut accumulator = out.at(&unit).accumulating(block, Semiring::SUM_PROD);
        accumulator.mm(&a.at(&unit), &b.at(&unit));
    }
}
