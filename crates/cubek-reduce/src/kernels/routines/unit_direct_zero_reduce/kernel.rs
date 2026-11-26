use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::zero::unit::UnitDirectZeroMicrokernel;

/// Every unit of the launch, zeroing its cells of `out`.
///
/// - `out`: one tensor, what a reduction writes.
#[allow(
    clippy::extra_unused_type_parameters,
    reason = "`Out` is the output's element, set by `#[define]`"
)]
#[cube(launch_unchecked)]
pub(crate) fn unit_direct_zero_reduce_kernel<Out: Numeric>(
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitDirectZeroMicrokernel,
    #[define(Out)] _element: ElemType,
) {
    let o = out.tile_mut(&partitioning);
    for unit in &partitioning {
        let mut cells = o.at(&unit);
        microkernel.runtime().zero(&mut cells);
    }
}
