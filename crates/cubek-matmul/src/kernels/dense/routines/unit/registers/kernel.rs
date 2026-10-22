use cubecl::tile::launch::Destination;
use cubecl::tile::{Partitioning, Semiring, TileArg};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::dense::unit::{
    UnitRegistersMicrokernel, unit_registers_microkernel,
};

/// Every plane of the launch, each of its units running its block, walked within the weight: a
/// batch whose entry of a table names no slab holds nothing, so its units take no step and write
/// zeros.
#[cube(launch_unchecked)]
pub fn unit_registers_matmul_kernel<EA: Numeric, E: Numeric, VO: Size, O: Destination>(
    lhs: &TileArg<'_>,
    rhs: &TileArg<'_>,
    out: &O::Arg<E>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitRegistersMicrokernel,
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
            for step in unit.walk_within(&b) {
                unit_registers_microkernel(&step, &sum, &a, &b);
            }
            sum.drained_into(&c);
        }
    }
}
