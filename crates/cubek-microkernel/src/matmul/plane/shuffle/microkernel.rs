use super::KStepOrder;
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::{Region, Tile};
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

/// The step `owner`, the owner of one run of columns, starts its walk of `total` steps of `K` at
/// under `order`: the first in order; rotated, `owner × stride` steps on, wrapped round the walk.
/// The walk then takes the steps from there to the last, then from the first up to it.
#[cube]
pub fn k_step_start(owner: usize, total: usize, #[comptime] order: KStepOrder) -> usize {
    match comptime!(order) {
        KStepOrder::InOrder => 0usize,
        KStepOrder::Rotated(stride) => {
            // A walk of no steps starts nowhere, and divides nothing.
            let steps = select(total == 0usize, 1usize, total);
            let start = (owner * comptime!(stride.get())) % steps;
            // Held inside the walk, so `total - start` never wraps: a walk of one constant step
            // was compiled with `% 1` dropped on ROCm, its start the owner itself, and its first
            // loop ran until the GPU hung.
            select(start < total, start, 0usize)
        }
    }
}
