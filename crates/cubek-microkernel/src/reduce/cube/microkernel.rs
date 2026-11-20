use super::base::CubeStagedReduceMicrokernel;
use crate::reduce::{
    ReduceAccumulator, ReduceCarrier, accumulate_strided, is_first_unit_of_cube, row_lanes,
    row_values,
};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's row: each of its units accumulates every `units`-th line of `input`, the cube's window of
/// the rows it reduces, consecutive units reading consecutive lines; each plane's lanes meet in the
/// plane's instructions, the planes meet in shared memory ([`ReduceAccumulator::across_planes`]), and the
/// cube's first unit writes the total to `out`, the cube's window of the output.
#[cube]
pub(crate) fn cube_staged_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    match microkernel.item().operation.carrier() {
        ReduceCarrier::Values => cube_staged_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::Indexed => cube_staged_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::TopK { k } => {
            cube_staged_reduce_top_k::<In, Acc, Out, Idx, N>(input, out, indices, microkernel, k)
        }
        ReduceCarrier::TopValues { k } => cube_staged_reduce_top_values::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            microkernel,
            k,
        ),
    }
}

#[cube]
fn cube_staged_reduce<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    let operation = microkernel.item().operation;
    let row_lanes = row_lanes(input);
    let count = input.flat::<In, N>().shape();

    let total = cube_staged_reduce_part::<In, Acc, N, C>(input, 0, count, microkernel);

    let values = row_values(count, row_lanes);

    if is_first_unit_of_cube() {
        C::write::<Out, Idx>(
            total,
            out,
            indices,
            values,
            row_lanes,
            operation,
            microkernel.item().writes(),
        );
    }
}

/// The first `count` lines of `input`, the line at `l` read as line `first_line + l` of its row,
/// accumulated by the cube into `C`: each unit takes every `units`-th line, consecutive units
/// reading consecutive lines, each plane's lanes meet in the plane's instructions and the planes in
/// shared memory ([`ReduceAccumulator::across_planes`]). The cube's first unit is left with the
/// total.
#[cube]
pub(crate) fn cube_staged_reduce_part<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    first_line: usize,
    count: usize,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) -> C {
    let monoid = microkernel.item().operation.monoid();
    let unit = PLANE_POS as usize * microkernel.lanes() + UNIT_POS_PLANE as usize;
    let acc = accumulate_strided::<In, Acc, N, C>(
        input,
        unit,
        microkernel.units(),
        count,
        first_line,
        microkernel.item(),
    );

    let plane = C::across(acc, UnitShare::Plane, monoid);
    C::across_planes(plane, microkernel.planes(), monoid)
}

/// [`cube_staged_reduce`] into a top-k of `k` slots, `K` bound in the function that uses it.
#[cube]
fn cube_staged_reduce_top_k<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    cube_staged_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(input, out, indices, microkernel)
}

/// [`cube_staged_reduce`] into a top-k of `k` values without their indices, `K` bound in the
/// function that uses it.
#[cube]
fn cube_staged_reduce_top_values<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    cube_staged_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
        input,
        out,
        indices,
        microkernel,
    )
}
