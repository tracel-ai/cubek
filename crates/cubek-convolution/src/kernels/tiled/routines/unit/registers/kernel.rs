use cubecl::tile::{Axis, Partitioning, RegisterBlock, Semiring, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};

/// Every plane of the launch, each of its units contracting its block of the output against the
/// gathered input and the rhs, read straight from global memory. The block is accumulated as a
/// memory window, whose leaf walks every contracted axis itself: a register accumulator cannot
/// step a gathered operand. A plane whose box starts past the output on any bounded axis does nothing.
#[cube(launch_unchecked)]
pub fn unit_registers_conv_kernel<EA: Numeric, EL: Numeric>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] block: RegisterBlock,
    #[comptime] bounds: Vec<(Axis, usize)>,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let a = input.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);

    for plane in &partitioning {
        let mut inside = true;
        #[unroll]
        for i in 0..comptime!(bounds.len()) {
            let (axis, extent) = comptime!(bounds[i]);
            inside = inside && plane.origin(axis) < extent;
        }
        if inside {
            for unit in plane {
                let mut c = c.at(&unit).accumulating(block, Semiring::SUM_PROD);
                c.mm(&a.at(&unit), &b.at(&unit));
            }
        }
    }
}
