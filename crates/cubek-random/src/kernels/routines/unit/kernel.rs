use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::random::unit::{UnitRandomMicrokernel, unit_random_microkernel};
use cubek_microkernel::random::{Distribution, DistributionArgs, Seeds};

type Args<D> = <D as DistributionArgs>::Args;

/// Every unit of the launch, filling its run of the output with draws of `D`.
#[cube(launch_unchecked)]
pub fn unit_random_kernel<D: Distribution, E: Numeric, N: Size>(
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    args: Args<D>,
    seeds: Seeds,
    #[comptime] microkernel: UnitRandomMicrokernel,
    #[define(E)] _element: ElemType,
) {
    let o = out.tile(&partitioning);
    for unit in &partitioning {
        let mut run = o.at(&unit);
        unit_random_microkernel::<D, E, N>(&mut run, &args, &seeds, microkernel);
    }
}
