use cubecl::tile::launch::Destination;
use cubecl::tile::{Instruction, Partitioning, Semiring, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::MatmulOperands;
use cubek_microkernel::matmul::plane::{PlaneShuffleFetch, plane_shuffle_walk};

/// Every plane of the launch, running its run of the output. Each input is read at the element it
/// is bound at, under the scales it carries, none where it is plain, which ride every read the
/// body makes, each step's loaded as `fetch` says. `O` is how the output is reached: a buffer,
/// or a sink.
///
/// The walk is within the `[k, n]` input: where it reads a table, a batch whose entry names no
/// slab holds nothing, so its plane takes no step and writes zeros; a plain one is walked whole.
#[cube(launch_unchecked)]
pub fn plane_shuffle_matmul_kernel<EA: Numeric, E: Numeric, VO: Size, O: Destination>(
    lhs: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] instruction: Instruction,
    #[comptime] fetch: PlaneShuffleFetch,
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
        plane_shuffle_walk(
            plane.walk_within(&b),
            &sum,
            &a,
            &b,
            MatmulOperands::AS_BOUND,
            fetch,
        );

        c.dequantize_from(&sum.scaled(&a_factor).scaled(&b_factor));
    }
    O::finish::<E>(out);
}
