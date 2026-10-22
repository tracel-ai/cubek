use crate::microkernel::dense::unit::{UnitWholeKMicrokernel, unit_whole_k_microkernel};
use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, Semiring, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};

/// Every plane of the launch, each of its units running its run of the output. `O` is how the output
/// is reached: a buffer, or a sink.
#[cube(launch_unchecked)]
pub fn unit_whole_k_matmul_kernel<EA: Numeric, E: Numeric, VO: Size, O: Destination>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitWholeKMicrokernel,
    #[define(EA, E)] _elements: [ElemType; 2],
) {
    let a = lhs.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = O::tile::<E, VO>(out, &partitioning);
    for plane in &partitioning {
        for unit in plane {
            let (c, a, b) = (c.at(&unit), a.at(&unit), b.at(&unit));
            let mut sum = c.accumulator::<EA>(
                &a,
                &b,
                comptime!(microkernel.tile_instruction()),
                Semiring::SUM_PROD,
            );
            sum.zero();
            for step in &unit {
                unit_whole_k_microkernel(&step, &sum, &a, &b);
            }
            sum.drained_into(&c);
        }
    }
}
