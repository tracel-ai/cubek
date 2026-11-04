use super::PlaneShuffleFetch;
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::{Region, Tile, Walk};
use cubecl::{self as cubecl, cube, prelude::*};

/// `sum += a · b` over one step of `plane`: each unit adds its block. The caller opens `sum` (an
/// `accumulator` of `Instruction::Registers` at
/// [`tile_instruction`](crate::matmul::plane::PlaneShuffleMicrokernel::tile_instruction)), walks
/// `K` and drains it; the drain sums the units' partials across the plane. Each unit reads its
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

/// `sum += a · b` over every step `steps` walks, each a step of
/// [`plane_shuffle_microkernel`], its loads issued as `fetch` says: in the step, or a step ahead,
/// each unit holding the next step's reads in registers while this one contracts
/// (`Tile::mma_prefetched`), which reads its operands as bound.
#[cube]
pub fn plane_shuffle_walk(
    steps: Walk,
    sum: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] reads: MatmulOperands<MatmulRead>,
    #[comptime] fetch: PlaneShuffleFetch,
) {
    match comptime!(fetch) {
        PlaneShuffleFetch::InStep => {
            for step in steps {
                plane_shuffle_microkernel(&step, sum, a, b, reads);
            }
        }
        PlaneShuffleFetch::StepAhead => {
            comptime!(assert!(
                reads == MatmulOperands::AS_BOUND,
                "plane_shuffle_walk: a step ahead holds its operands' reads as bound"
            ));
            sum.mma_prefetched(&steps, a, b);
        }
    }
}
