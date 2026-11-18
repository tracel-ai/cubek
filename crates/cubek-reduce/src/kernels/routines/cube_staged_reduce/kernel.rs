use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::cube::CubeStagedReduceMicrokernel;

/// Every cube of the launch, reducing its rows into the carrier its operation accumulates into.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
#[cube(launch_unchecked)]
pub(crate) fn cube_staged_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    for cube in &partitioning {
        let mut rows_out = o.at(&cube);
        let rows = x.at(&cube);
        microkernel.runtime().execute::<In, Acc, Out, Out, N>(
            &rows,
            &mut rows_out,
            &ComptimeOption::new_None(),
        );
    }
}

/// Every cube of the launch, reducing its rows into the carrier its operation accumulates into, and
/// writing the values to `out` and their indices to `indices`, laid out as `out`.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `indices`: one tensor laid out as `out`.
#[cube(launch_unchecked)]
pub(crate) fn cube_staged_reduce_with_indices_kernel<
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
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile(&partitioning);
    let i = indices.tile(&partitioning);
    for cube in &partitioning {
        let mut rows_out = o.at(&cube);
        let rows = x.at(&cube);
        microkernel.runtime().execute::<In, Acc, Out, Idx, N>(
            &rows,
            &mut rows_out,
            &ComptimeOption::new_Some(i.at(&cube)),
        );
    }
}
