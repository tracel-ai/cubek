use cubecl::tile::{Partitioning, Semiring, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::matmul::unit::{UnitRegistersMicrokernel, unit_registers_microkernel};

/// Every plane of the launch, each of its units running its block of the output against the
/// gathered input and the rhs, read straight from global memory through every contracted cell.
#[cube(launch_unchecked)]
pub fn unit_registers_conv_kernel<EA: Numeric, EL: Numeric>(
    input: &TileTensor<'_>,
    rhs: &TileTensor<'_>,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitRegistersMicrokernel,
    #[define(EA, EL)] _elements: [ElemType; 2],
) {
    let a = input.tile(&partitioning);
    let b = rhs.tile(&partitioning);
    let c = out.tile(&partitioning);

    for plane in &partitioning {
        for unit in plane {
            let (c, a, b) = (c.at(&unit), a.at(&unit), b.at(&unit));
            let mut sum = c.accumulator::<EA>(
                &a,
                &b,
                microkernel.drained_instruction(),
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
