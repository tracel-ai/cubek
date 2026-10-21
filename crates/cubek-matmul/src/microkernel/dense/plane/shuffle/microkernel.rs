use crate::microkernel::{MatmulOperands, MatmulRead};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_tile::{Region, Tile};

/// `sum += a · b` over one step of `plane`: each unit adds its block. The caller opens `sum`
/// (an `accumulator` of `Instruction::Registers` at [`tile_instruction`](crate::microkernel::dense::plane::PlaneShuffleMicrokernel::tile_instruction)), walks `K`
/// and drains it; the drain folds the units' partials across the plane. Each unit reads its
/// window of each operand as `reads` says.
#[cube]
pub fn plane_shuffle_microkernel(
    step: &Region,
    sum: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] reads: MatmulOperands<MatmulRead>,
) {
    for unit in step {
        let mut unit_sum = sum.at(&unit);
        let b_unit = MatmulRead::leaf(&unit_sum, &b.at(&unit), reads.rhs);
        let a_unit = MatmulRead::leaf(&unit_sum, &a.at(&unit), reads.lhs);
        unit_sum.mma(&a_unit, &b_unit);
    }
}
