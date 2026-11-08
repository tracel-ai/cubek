use cubecl::tile::launch::Destination;
use cubecl::tile::{Instruction, Partitioning, Semiring, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::plane::{KStepOrder, k_step_start, plane_shuffle_microkernel};

/// Every plane of the launch, running its run of the output. Each input is read at the element it
/// is bound at, under the scales it carries, none where it is plain, which ride every read the
/// body makes. `O` is how the output is reached: a buffer, or a sink.
///
/// The walk is within the `[k, n]` input: where it reads a table, a batch whose entry names no
/// slab holds nothing, so its plane takes no step and writes zeros; a plain one is walked whole.
/// Each plane walks `K` in `k_order`, counted among the launch's planes.
#[cube(launch_unchecked)]
pub fn plane_shuffle_matmul_kernel<EA: Numeric, E: Numeric, VO: Size, O: Destination>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] instruction: Instruction,
    #[comptime] k_order: KStepOrder,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);

    for plane in &partitioning {
        let (mut c, a, b) = (c.at(&plane), a.at(&plane), b.at(&plane));
        let (a, a_factor) = a.factor(c.space(), instruction);
        let (b, b_factor) = b.factor(c.space(), instruction);

        let mut sum = c.accumulator::<EA>(&a, &b, instruction, Semiring::SUM_PROD);

        sum.zero();
        match comptime!(k_order) {
            KStepOrder::InOrder => {
                for k_step in plane.walk_within(&b) {
                    plane_shuffle_microkernel(&k_step, &sum, &a, &b, MatmulOperands::AS_BOUND);
                }
            }
            KStepOrder::Rotated(_) => {
                let total = plane.walk_within(&b).total();
                let owner = CUBE_POS * CUBE_DIM_Y as usize + UNIT_POS_Y as usize;
                let start = k_step_start(owner, total, k_order);
                for k_step in plane.walk_within(&b).range(start, total - start) {
                    plane_shuffle_microkernel(&k_step, &sum, &a, &b, MatmulOperands::AS_BOUND);
                }
                for k_step in plane.walk_within(&b).range(0, start) {
                    plane_shuffle_microkernel(&k_step, &sum, &a, &b, MatmulOperands::AS_BOUND);
                }
            }
        }

        c.dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));
    }
}
