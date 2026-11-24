use cubecl::ir::ElemType;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{AccumulateArg, TileOperand};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernel;

/// Every cube of the launch, reducing its part of its rows as the cube microkernel reduces a row,
/// its total added into `out`, which holds zero before the launch.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide, at zero.
#[cube(launch_unchecked)]
pub(crate) fn hypercube_atomic_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut AccumulateArg<'_, Out>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    for cube in &partitioning {
        microkernel.runtime().reduce::<In, Acc, Out, Out, N>(
            &cube,
            &x,
            &o,
            &ComptimeOption::new_None(),
        );
    }
}
