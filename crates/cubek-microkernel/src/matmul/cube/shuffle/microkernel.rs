use super::base::CubeMeet;
use crate::matmul::plane::{KStepOrder, k_step_start, plane_shuffle_microkernel};
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::{Axis, Instruction, Region, Semiring, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's run of the output: `c = a · b` over `cube`, its planes each summing their slice of
/// every step in `EA`, adding their partials into one accumulator in shared memory as `meet` says,
/// and the sum copied into `c` once the cube has synchronized, so a half-precision cell rounds
/// once. What `factor` takes out of the operands, nothing for operands decoded as they are read,
/// is applied as each plane adds its partial.
///
/// Two levels under the cube: its `planes` planes, each walking its run of the steps along
/// `split`, then their units. Each plane's drain merges its units' partials across the plane,
/// then adds into the accumulator. The units read each operand as `reads` says. Each plane walks
/// its steps in `k_order`, counted among the launch's cubes: a cube's planes split `K` between
/// them, so it is the cubes side by side whose planes read alike.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_shuffle_microkernel<EA: Numeric>(
    cube: &Region,
    c: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] instruction: Instruction,
    #[comptime] meet: CubeMeet,
    #[comptime] split: Axis,
    #[comptime] planes: usize,
    #[comptime] reads: MatmulOperands<MatmulRead>,
    #[comptime] k_order: KStepOrder,
) {
    let mut c = c.clone();
    let (a, a_factor) = a.factor(c.space(), instruction);
    let (b, b_factor) = b.factor(c.space(), instruction);

    match meet {
        CubeMeet::Atomic => {
            let mut accumulator = c.smem_accumulation::<EA>();
            let mut sum =
                accumulator
                    .sink
                    .accumulator::<EA>(&a, &b, instruction, Semiring::SUM_PROD);
            sum.zero();

            walk_steps(cube, &sum, &a, &b, reads, k_order);

            accumulator
                .sink
                .dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));

            sync_cube();

            c.copy_from(&accumulator.source);
        }
        CubeMeet::InSlots => {
            let accumulator = c.smem_slots_accumulation::<EA>(planes);
            let mut sum = c.accumulator::<EA>(&a, &b, instruction, Semiring::SUM_PROD);
            sum.zero();

            // A plane's place along the split, which is its slot among the writers.
            let writer = cube.walk().position(split);
            walk_steps(cube, &sum, &a, &b, reads, k_order);

            accumulator.drain(&sum.scaled(&a_factor).scaled(&b_factor), writer);
            c.copy_from(&accumulator.source);
        }
    }
}

/// `sum += a · b` over every step of `cube` this plane walks, in `k_order`, the cube its owner.
#[cube]
fn walk_steps(
    cube: &Region,
    sum: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] reads: MatmulOperands<MatmulRead>,
    #[comptime] k_order: KStepOrder,
) {
    match comptime!(k_order) {
        KStepOrder::InOrder => {
            for step in cube {
                plane_shuffle_microkernel(&step, sum, a, b, reads);
            }
        }
        KStepOrder::Rotated(_) => {
            let total = cube.walk().total();
            let start = k_step_start(CUBE_POS, total, k_order);
            for step in cube.walk().range(start, total - start) {
                plane_shuffle_microkernel(&step, sum, a, b, reads);
            }
            for step in cube.walk().range(0, start) {
                plane_shuffle_microkernel(&step, sum, a, b, reads);
            }
        }
    }
}
