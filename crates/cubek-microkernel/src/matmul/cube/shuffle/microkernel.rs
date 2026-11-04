use super::base::CubeMeet;
use crate::matmul::plane::{PlaneShuffleFetch, plane_shuffle_walk};
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
/// then adds into the accumulator. The units read each operand as `reads` says, and issue each
/// step's loads as `fetch` says.
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
    #[comptime] fetch: PlaneShuffleFetch,
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

            plane_shuffle_walk(cube.walk(), &sum, &a, &b, reads, fetch);

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

            let steps = cube.walk();

            // A plane's place along the split, which is its slot among the writers.
            let writer = steps.position(split);
            plane_shuffle_walk(steps, &sum, &a, &b, reads, fetch);

            accumulator.drain(&sum.scaled(&a_factor).scaled(&b_factor), writer);
            c.copy_from(&accumulator.source);
        }
    }
}
