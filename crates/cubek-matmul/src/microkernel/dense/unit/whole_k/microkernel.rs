use cubecl::{self as cubecl, cube, prelude::*};
use cubek_tile::{Region, Tile};

/// `sum += a · b` over one step of a unit's block. The caller opens `sum` (an `accumulator` of
/// `Instruction::Registers` at [`tile_instruction`](crate::microkernel::dense::unit::UnitWholeKMicrokernel::tile_instruction)),
/// walks `K` and drains it.
#[cube]
pub fn unit_whole_k_microkernel(step: &Region, sum: &Tile, a: &Tile, b: &Tile) {
    let mut sum_step = sum.at(step);
    sum_step.mma(&a.at(step), &b.at(step));
}
