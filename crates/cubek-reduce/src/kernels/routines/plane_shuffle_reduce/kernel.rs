use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::plane::PlaneShuffleReduceMicrokernel;

/// Every plane of the launch, reducing its rows into the carrier its operation accumulates into.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
#[cube(launch_unchecked)]
pub(crate) fn plane_shuffle_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[define(In, Acc, Out)] _elements: [ElemType; 3],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    for plane in &partitioning {
        microkernel.runtime().reduce::<In, Acc, Out, Out, N>(
            &plane,
            &x,
            &o,
            &ComptimeOption::new_None(),
        );
    }
}

/// Every plane of the launch, reducing its rows into the carrier its operation accumulates into, and
/// writing the values to `out` and their indices to `indices`, laid out as `out`.
///
/// Its operands' leaves:
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `indices`: one tensor laid out as `out`.
#[cube(launch_unchecked)]
pub(crate) fn plane_shuffle_reduce_with_indices_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut TileTensor<'_>,
    indices: &mut TileTensor<'_>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    let i = indices.tile_mut(&partitioning);
    let indices = ComptimeOption::new_Some(i);
    for plane in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Idx, N>(&plane, &x, &o, &indices);
    }
}
