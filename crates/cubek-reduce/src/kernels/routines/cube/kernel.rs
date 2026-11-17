use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::cube::{CubeReduceMicrokernel, cube_reduce_microkernel};

/// Every cube of the launch, reducing its rows into the carrier its operation accumulates into.
#[cube(launch_unchecked)]
pub fn cube_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
    O: TileOperand,
>(
    input: &I::Arg,
    out: &O::Arg,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = O::tile::<Out>(out, &partitioning);
    for cube in &partitioning {
        let mut rows_out = o.at(&cube);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&cube);
        let rows = x.at(&cube);
        cube_reduce_microkernel::<In, Acc, Out, Out, N>(
            &rows,
            &mut rows_out,
            &mut unwritten,
            microkernel,
        );
    }
}

/// Every cube of the launch, reducing its rows into the carrier its operation accumulates into, and
/// writing the values to `out` and their indices to `indices`, laid out as `out`.
#[cube(launch_unchecked)]
pub fn cube_reduce_with_indices_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    I: TileOperand,
    O: TileOperand,
>(
    input: &I::Arg,
    out: &O::Arg,
    indices: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = O::tile::<Out>(out, &partitioning);
    let i = indices.tile(&partitioning);
    for cube in &partitioning {
        let mut rows_out = o.at(&cube);
        let mut rows_indices = i.at(&cube);
        let rows = x.at(&cube);
        cube_reduce_microkernel::<In, Acc, Out, Idx, N>(
            &rows,
            &mut rows_out,
            &mut rows_indices,
            microkernel,
        );
    }
}
