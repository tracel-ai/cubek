use cubecl::ir::ElemType;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{AccumulateArg, TileOperand};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::hypercube::HypercubeAtomicReduceMicrokernel;

/// Every cube of the launch, its part of its rows added into `out`.
///
/// - `input`: the rows, as every reduce kernel reads them.
/// - `out`: one tensor of a cell a row, at zero before the launch, which only receives atomic adds.
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
    #[comptime] microkernel: HypercubeAtomicReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    for cube in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, N, Vector<Acc, N>>(&cube, &x, &o);
    }
}
