use cubecl::ir::ElemType;
use cubecl::tile::Partitioning;
use cubecl::tile::launch::{AccumulateArg, TileOperand};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::cube::{CubeReduceMicrokernel, cube_reduce_microkernel};

/// Every cube of the launch, reducing its part of its rows as the cube microkernel reduces a row,
/// its total added into `out`, which holds zero before the launch.
#[cube(launch_unchecked)]
pub fn hypercube_atomic_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &AccumulateArg<'_, Out>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    for cube in &partitioning {
        let mut rows_out = o.at(&cube);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&cube);
        let part = x.at(&cube);
        cube_reduce_microkernel::<In, Acc, Out, Out, N>(
            &part,
            &mut rows_out,
            &mut unwritten,
            microkernel,
        );
    }
}
