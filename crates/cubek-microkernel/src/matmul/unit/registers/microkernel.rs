use cubecl::tile::{Region, Semiring, SoftwareMatmul, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// `sum += a · b` over one step of a unit's block. The caller opens `sum` (a
/// `register_accumulator` of the [`block`](crate::matmul::unit::UnitRegistersMicrokernel::block)),
/// walks
/// `K` and drains it.
#[cube]
pub fn unit_registers_microkernel(step: &Region, sum: &Tile, a: &Tile, b: &Tile) {
    let mut sum_step = sum.at(step);
    sum_step.mma(&a.at(step), &b.at(step));
}

/// Each unit of `plane` accumulating `out += a · b` over its block as a memory window, which the
/// leaf walks along every contracted axis itself: the way a block whose operand a register
/// accumulator cannot step (a gathered one) is contracted.
#[cube]
pub fn unit_window_microkernel(
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
