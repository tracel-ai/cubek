use super::base::CubeMeet;
use crate::matmul::plane::plane_shuffle_microkernel;
use crate::matmul::{MatmulOperands, MatmulRead};
use cubecl::tile::launch::{ServedDestination, ServedDestinationExpand};
use cubecl::tile::{Axis, Region, Semiring, SoftwareMatmul, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's run of the output: `c = a · b` over `cube`, its planes each summing their slice of
/// every step in `EA`, adding their partials into one accumulator in shared memory as `meet` says,
/// and the total handed to `dest` at `cube` once the cube has synchronized, so a half-precision
/// cell rounds once. What `factor` takes out of the operands, nothing for operands decoded as they are read,
/// is applied as each plane adds its partial.
///
/// Two levels under the cube: its `planes` planes, each walking its run of the steps along
/// `split`, then their units. Each plane's drain merges its units' partials across the plane,
/// then adds into the accumulator. The units read each operand as `reads` says.
#[cube]
pub fn cube_shuffle_microkernel<EA: Numeric, D: ServedDestination>(
    cube: &Region,
    dest: &D,
    a: &Tile,
    b: &Tile,
    #[comptime] block: Option<SoftwareMatmul>,
    #[comptime] meet: CubeMeet,
    #[comptime] split: Axis,
    #[comptime] planes: usize,
    #[comptime] reads: MatmulOperands<MatmulRead>,
) {
    let c = dest.tile().at(cube);

    match meet {
        CubeMeet::Atomic => {
            let mut accumulator = c.smem_accumulation::<EA>();
            let mut sum = open_sum::<EA>(&accumulator.sink, a, b, block);
            let (a, a_factor) = a.factor(&sum);
            let (b, b_factor) = b.factor(&sum);
            sum.zero();

            for step in cube {
                plane_shuffle_microkernel(&step, &sum, &a, &b, reads);
            }

            accumulator
                .sink
                .dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));

            sync_cube();

            accumulator.source.consumed_by::<D>(dest, cube);
        }
        CubeMeet::InSlots => {
            let accumulator = c.smem_slots_accumulation::<EA>(planes);
            let mut sum = open_sum::<EA>(&c, a, b, block);
            let (a, a_factor) = a.factor(&sum);
            let (b, b_factor) = b.factor(&sum);
            sum.zero();

            let steps = cube.walk();

            // A plane's place along the split, which is its slot among the writers.
            let writer = steps.position(split);
            for step in steps {
                plane_shuffle_microkernel(&step, &sum, &a, &b, reads);
            }

            accumulator.drain(&sum.scaled(&a_factor).scaled(&b_factor), writer);
            accumulator.source.consumed_by::<D>(dest, cube);
        }
    }
}

/// The sum `over` opens for `a · b`: the register block `block` names, or the plane's fragments
/// where it names none.
#[cube]
fn open_sum<EA: Numeric>(
    over: &Tile,
    a: &Tile,
    b: &Tile,
    #[comptime] block: Option<SoftwareMatmul>,
) -> Tile {
    match block {
        Some(block) => over.register_accumulator::<EA>(a, b, block, Semiring::SUM_PROD),
        None => over.accumulator::<EA>(a, b, Semiring::SUM_PROD),
    }
}
