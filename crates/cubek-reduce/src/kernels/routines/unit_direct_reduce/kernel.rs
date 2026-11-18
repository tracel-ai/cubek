use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::unit::UnitDirectReduceMicrokernel;

/// Every unit of the launch, reducing its rows into the carrier its operation accumulates into.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
#[cube(launch_unchecked)]
pub(crate) fn unit_direct_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    for unit in &partitioning {
        let mut rows_out = o.at(&unit);
        // One output: the indices are never written, so they get the output's own window.
        let mut unwritten = o.at(&unit);
        let rows = x.at(&unit);
        microkernel
            .runtime()
            .execute::<In, Acc, Out, Out, N>(&rows, &mut rows_out, &mut unwritten);
    }
}

/// Every unit of the launch, reducing its rows into the carrier its operation accumulates into, and
/// writing the values to `out` and their indices to `indices`, laid out as `out`.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `indices`: one tensor laid out as `out`.
#[cube(launch_unchecked)]
pub(crate) fn unit_direct_reduce_with_indices_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &TileTensor<'_>,
    indices: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: UnitDirectReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    let i = indices.tile(&partitioning);
    for unit in &partitioning {
        let mut rows_out = o.at(&unit);
        let mut rows_indices = i.at(&unit);
        let rows = x.at(&unit);
        microkernel.runtime().execute::<In, Acc, Out, Idx, N>(
            &rows,
            &mut rows_out,
            &mut rows_indices,
        );
    }
}
