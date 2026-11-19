use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::zero::unit::unit_zero_microkernel;

/// Every unit of the launch, zeroing its cells of `out`.
#[cube(launch_unchecked)]
pub fn unit_zero_kernel<Out: Numeric>(
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[define(Out)] _element: ElemType,
) {
    let o = out.tile_mut(&partitioning);
    for unit in &partitioning {
        let mut cells = o.at(&unit);
        unit_zero_microkernel(&mut cells);
    }
}
