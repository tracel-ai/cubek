use super::base::CubeMeet;
use crate::matmul::MatmulContraction;
use crate::matmul::plane::plane_shuffle_microkernel;
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::{Axis, Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's run of the output: `c = a · b` over `cube`, its planes each summing their slice of
/// every step in `EA`, adding their partials into one accumulator in shared memory as `meet` says,
/// and the sum copied into `c` once the cube has synchronized, so a half-precision cell rounds
/// once. What `factor` takes out of the operands, nothing for operands decoded as they are read,
/// is applied as each plane adds its partial.
///
/// Two levels under the cube: its `planes` planes, each walking its run of the steps along
/// `split`, then their units. Each plane's drain merges its units' partials across the plane,
/// then adds into the accumulator. The units read each operand as `reads` says; the sum is opened
/// from the planes' `contraction`, on the factor their instruction lays its rows along.
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn cube_shuffle_microkernel<EA: Numeric>(
    cube: &Region,
    c: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] contraction: MatmulContraction,
    #[comptime] meet: CubeMeet,
    #[comptime] split: Axis,
    #[comptime] planes: usize,
    #[comptime] reads: MatmulOperands<MatmulRead>,
) {
    let mut c = c.clone();
    let (a, a_factor) = a.factor(c.space(), contraction.instruction);
    let (b, b_factor) = b.factor(c.space(), contraction.instruction);

    match meet {
        CubeMeet::Atomic => {
            let mut accumulator = c.smem_accumulation::<EA>();
            let mut sum =
                MatmulContraction::accumulator::<EA>(&accumulator.sink, &a, &b, contraction);
            sum.zero();

            for step in cube {
                plane_shuffle_microkernel(&step, &sum, &a, &b, reads);
            }

            accumulator
                .sink
                .dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));
            sync_cube();

            c.copy_from(&accumulator.source);
        }
        CubeMeet::InSlots => {
            let accumulator = c.smem_slots_accumulation::<EA>(planes);
            let mut sum = MatmulContraction::accumulator::<EA>(&c, &a, &b, contraction);
            sum.zero();

            let steps = cube.walk();

            // A plane's place along the split, which is its slot among the writers.
            let writer = steps.position(split);
            for step in steps {
                plane_shuffle_microkernel(&step, &sum, &a, &b, reads);
            }

            accumulator.drain(&sum.scaled(&a_factor).scaled(&b_factor), writer);
            c.copy_from(&accumulator.source);
        }
    }
}
