use cubecl::tile::{Axis, Partitioning, RegisterBlock, Semiring, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};

/// Every plane of the launch, each of its units contracting its block of the output against the
/// gathered input and the rhs, read straight from global memory. The block is accumulated as a
/// memory window, whose leaf walks every contracted axis itself: a register accumulator cannot
/// step a gathered operand. A plane whose box starts past the output on any bounded axis does
/// nothing.
#[cube(launch_unchecked)]
pub fn unit_registers_conv_kernel<EA: Numeric, EL: Numeric>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] block: RegisterBlock,
    #[comptime] bounded: Vec<Axis>,
    extents: Sequence<usize>,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let input = input.tile(&partitioning);
    let rhs = rhs.tile(&partitioning);
    let out = out.tile(&partitioning);

    for plane in &partitioning {
        let mut inside = true;
        #[unroll]
        for i in 0..comptime!(bounded.len()) {
            let axis = comptime!(bounded[i]);
            inside = inside && plane.origin(axis) < *extents.index(i);
        }
        if inside {
            for unit in plane {
                let mut accumulator = out.at(&unit).accumulating(block, Semiring::SUM_PROD);
                accumulator.mm(&input.at(&unit), &rhs.at(&unit));
            }
        }
    }
}
