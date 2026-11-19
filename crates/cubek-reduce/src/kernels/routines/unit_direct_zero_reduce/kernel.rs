use cubecl::ir::ElemType;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::zero::unit::UnitDirectZeroMicrokernel;

/// Every unit of the launch, zeroing its cells of `out`.
///
/// Its operands' leaves:
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
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
        microkernel.runtime().execute(&mut cells);
    }
}
