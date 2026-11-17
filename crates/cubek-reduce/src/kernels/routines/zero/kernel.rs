use cubecl::ir::ElemType;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::TileOperand;
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::zero::unit::unit_zero_microkernel;

/// Every unit of the launch, zeroing its cells of `out`.
///
/// Its operands' leaves:
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k), or
///   what the caller binds in its place (`O`).
#[cube(launch_unchecked)]
pub fn unit_zero_kernel<Out: Numeric, O: TileOperand>(
    out: &O::Arg,
    partitioning: Partitioning,
    #[define(Out)] _element: ElemType,
) {
    let o = O::tile::<Out>(out, &partitioning);
    for unit in &partitioning {
        let mut cells = o.at(&unit);
        unit_zero_microkernel(&mut cells);
    }
}
