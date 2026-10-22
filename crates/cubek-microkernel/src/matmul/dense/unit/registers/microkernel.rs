use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// `sum += a · b` over one step of a unit's block. The caller opens `sum` (`block_accumulator` at
/// [`tile_instruction`](crate::matmul::dense::unit::UnitRegistersMicrokernel::tile_instruction)), walks `K` and drains it.
#[cube]
pub fn unit_registers_microkernel(step: &Region, sum: &Tile, a: &Tile, b: &Tile) {
    let mut sum_step = sum.at(step);
    sum_step.mma(&a.at(step), &b.at(step));
}
