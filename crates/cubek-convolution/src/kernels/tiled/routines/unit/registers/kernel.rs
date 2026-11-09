use cubecl::tile::{Partitioning, RegisterBlock, Semiring, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};

/// Every plane of the launch, each of its units contracting its block of the output against the
/// gathered input and the rhs, read straight from global memory. The block is accumulated as a
/// memory window, whose leaf walks every contracted axis itself: a register accumulator cannot
/// step a gathered operand.
#[cube(launch_unchecked)]
pub fn unit_registers_conv_kernel<EA: Numeric, EL: Numeric>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] block: RegisterBlock,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let a = input.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);

    for plane in &partitioning {
        for unit in plane {
            let mut c = c.at(&unit).accumulating(block, Semiring::SUM_PROD);
            c.mm(&a.at(&unit), &b.at(&unit));
        }
    }
}
