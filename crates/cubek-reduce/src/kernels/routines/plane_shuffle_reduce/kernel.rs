use cubecl::ir::ElemType;
use cubecl::tile::launch::TileOperand;
use cubecl::tile::{Partitioning, Tile, TileTensor};
use cubecl::{self as cubecl, cube, prelude::*};
use cubek_microkernel::reduce::ReduceCarrierFamily;
use cubek_microkernel::reduce::plane::PlaneShuffleReduceMicrokernel;

/// Every plane of the launch, reducing its rows into the carrier its operation accumulates into, and
/// writing their indices beside them where the launch binds any.
///
/// - `input`: the rows, one tensor of any rank read at its own strides, or what the caller binds in
///   its place (`I`).
/// - `out`: one tensor of the input's shape, the reduced dim one cell wide (`k` for a top-k).
/// - `indices`: one tensor of `out`'s shape at its own strides, where the launch writes the indices.
#[cube(launch_unchecked)]
pub(crate) fn plane_shuffle_reduce_kernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    K: Size,
    F: ReduceCarrierFamily,
    I: TileOperand,
>(
    input: &I::Arg,
    out: &mut TileTensor<'_>,
    indices: ComptimeOption<&mut TileTensor<'static>>,
    partitioning: Partitioning,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[define(In, Acc, Out, Idx)] _elements: [ElemType; 4],
) {
    let x = I::tile::<In>(input, &partitioning);
    let o = out.tile_mut(&partitioning);
    let i: ComptimeOption<Tile> = indices.map(|indices| indices.tile_mut(&partitioning));
    for plane in &partitioning {
        microkernel
            .runtime()
            .reduce::<In, Acc, Out, Idx, N, F::Carrier<Acc, N, K>>(&plane, &x, &o, &i);
    }
}
