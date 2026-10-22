use cubecl::tile::launch::Destination;
use cubecl::tile::{Instruction, Partitioning, Semiring, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::dense::plane::plane_shuffle_microkernel;
use cubek_microkernel::matmul::{MatmulOperands, MatmulRead};

/// Every plane of the launch, running its run of the output. The weight is read at the element it
/// is bound at, under the scales it carries, none where it is plain, which ride every read the
/// body makes. `O` is how the output is reached: a buffer, or a sink.
///
/// The walk is within the weight: where it reads a table, a batch whose entry names no slab holds
/// nothing, so its plane takes no step and writes zeros; a plain weight is walked whole.
#[cube(launch_unchecked)]
pub fn plane_shuffle_matmul_kernel<EA: Numeric, E: Numeric, VO: Size, O: Destination>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] instruction: Instruction,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);
    for plane in &partitioning {
        let (mut c, a, b) = (c.at(&plane), a.at(&plane), b.at(&plane));
        let (b, b_factor) = b.factor(c.space(), instruction);
        let mut sum = c.accumulator::<EA>(&a, &b, instruction, Semiring::SUM_PROD);
        sum.zero();
        for k_step in plane.walk_within(&b) {
            plane_shuffle_microkernel(
                &k_step,
                &sum,
                &a,
                &b,
                comptime!(MatmulOperands::all(MatmulRead::AsBound)),
            );
        }
        c.dequantize_from(&sum.scaled(&b_factor));
    }
}
