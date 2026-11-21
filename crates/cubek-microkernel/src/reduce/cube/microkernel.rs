use super::base::CubeStagedReduceMicrokernel;
use crate::reduce::{
    ReduceAccumulator, ReduceCarrier, accumulate_across, accumulate_strided, is_first_unit_of_cube,
    row_lanes, row_values, values_across, write_across,
};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Region, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's rows, `cube`'s region of `input`, reduced into `out` and `indices` at the same
/// region. Read one line of rows wide, each of its units accumulates every `units`-th line of the
/// cube's window, consecutive units reading consecutive lines; each plane's lanes meet in the
/// plane's instructions, the planes meet in shared memory
/// ([`ReduceAccumulator::across_planes`]), and the cube's first unit writes the total. Read
/// [`lines_across`](CubeStagedReduceMicrokernel::lines_across) lines of rows wide, each column's
/// units, in every plane, take its lines in turns as `cube` deals them, so one load reads whole
/// transactions; they meet within their plane's column, then the planes' columns in shared memory.
#[cube]
pub(crate) fn cube_staged_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    match microkernel.item().operation.carrier() {
        ReduceCarrier::Values => cube_staged_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
            cube,
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::Indexed => cube_staged_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
            cube,
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::TopK { k } => cube_staged_reduce_top_k::<In, Acc, Out, Idx, N>(
            cube,
            input,
            out,
            indices,
            microkernel,
            k,
        ),
        ReduceCarrier::TopValues { k } => cube_staged_reduce_top_values::<In, Acc, Out, Idx, N>(
            cube,
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
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    if microkernel.lines_across() == 1 {
        cube_staged_reduce_window::<In, Acc, Out, Idx, N, C>(
            cube,
            input,
            out,
            indices,
            microkernel,
        );
    } else {
        cube_staged_reduce_across::<In, Acc, Out, Idx, N, C>(
            cube,
            input,
            out,
            indices,
            microkernel,
        );
    }
}

/// [`cube_staged_reduce_microkernel`] read one line of rows wide.
#[cube]
fn cube_staged_reduce_window<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    let rows = input.at(cube);
    let mut rows_out = out.at(cube);
    let rows_indices = indices.clone().map(|tile| tile.at(cube));
    let operation = microkernel.item().operation;
    let row_lanes = row_lanes(&rows);
    let count = rows.flat::<In, N>().shape();

    let total = cube_staged_reduce_part::<In, Acc, N, C>(&rows, 0, count, microkernel);

    let values = row_values(count, row_lanes);

    if is_first_unit_of_cube() {
        C::write::<Out, Idx>(
            total,
            &mut rows_out,
            &rows_indices,
            values,
            row_lanes,
            operation,
            microkernel.item().writes(),
        );
    }
}

/// [`cube_staged_reduce_microkernel`] read `lines_across` lines of rows wide. The units of a
/// column hold the low bits of each plane's lane, as `cube`'s units level lays the reduced
/// dimension innermost, so a column meets across those bits alone, and its unit at the run's
/// start stages the plane's partial.
#[cube]
fn cube_staged_reduce_across<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    let monoid = microkernel.item().operation.monoid();
    let columns = microkernel.lines_across();
    let per_column = microkernel.lanes() / columns;
    let per_step = microkernel.units() / columns;
    let values = values_across::<In, N>(cube, input, columns);
    let acc = accumulate_across::<In, Acc, N, C>(cube, input, values, per_step, microkernel.item());
    let column_units = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    let plane = C::across(acc, column_units, monoid);
    let lane = UNIT_POS_PLANE as usize;
    let total = C::across_planes(
        plane,
        lane / per_column,
        lane.is_multiple_of(per_column),
        microkernel.planes(),
        columns,
        monoid,
    );
    write_across::<Out, Idx, Acc, N, C>(cube, total, out, indices, values, microkernel.item());
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
    let one_column = comptime!(1usize);
    C::across_planes(
        plane,
        0,
        UNIT_POS_PLANE == 0,
        microkernel.planes(),
        one_column,
        monoid,
    )
}

/// [`cube_staged_reduce_microkernel`] into a top-k of `k` slots, `K` bound in the
/// function that uses it.
#[cube]
fn cube_staged_reduce_top_k<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    cube_staged_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(
        cube,
        input,
        out,
        indices,
        microkernel,
    )
}

/// [`cube_staged_reduce_microkernel`] into a top-k of `k` values without their indices, `K`
/// bound in the function that uses it.
#[cube]
fn cube_staged_reduce_top_values<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    cube: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    cube_staged_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
        cube,
        input,
        out,
        indices,
        microkernel,
    )
}
