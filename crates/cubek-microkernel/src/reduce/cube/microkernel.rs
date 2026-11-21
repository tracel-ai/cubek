use super::base::CubeStagedReduceMicrokernel;
use crate::reduce::read::{
    UnitStridedReduceMicrokernel, UnitWalkReduceMicrokernel, unit_strided_reduce_microkernel,
    unit_walk_reduce_microkernel,
};
use crate::reduce::write::{
    UnitCarrierWriteMicrokernel, UnitWalkWriteMicrokernel, unit_carrier_write_microkernel,
    unit_walk_write_microkernel,
};
use crate::reduce::{
    REDUCED, ReduceAccumulator, ReduceCarrier, across_column, is_first_unit_of_cube,
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
/// units, in every plane, take its lines in turns as `cube`'s walk gives them, so one load reads whole
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

/// [`cube_staged_reduce_microkernel`] into the carrier `C`.
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
    let item = microkernel.item();
    if microkernel.lines_across() == 1 {
        // One line of rows wide: the cube reduces its window, and its first unit writes.
        let rows = input.at(cube);
        let count = rows.flat::<In, N>().shape();
        let total = cube_staged_reduce_part::<In, Acc, N, C>(&rows, 0, count, microkernel);

        if is_first_unit_of_cube() {
            let load = rows.vector_tile();
            let row_lanes = comptime!(load.extent_along(REDUCED));
            let mut rows_out = out.at(cube);
            let rows_indices = indices.clone().map(|tile| tile.at(cube));
            unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
                total,
                &mut rows_out,
                &rows_indices,
                count * row_lanes,
                comptime!(UnitCarrierWriteMicrokernel::new(item, row_lanes)),
            );
        }
    } else {
        // `lines_across` lines of rows wide: the units of a column hold the low bits of each
        // plane's lane, as `cube`'s units level lays the reduced dimension innermost, so a column
        // meets across those bits alone, and its unit at the run's start stages the plane's
        // partial.
        let monoid = item.operation.monoid();
        let columns = microkernel.lines_across();
        let per_column = microkernel.lanes() / columns;
        let per_step = microkernel.units() / columns;
        let window = input.at(cube);
        let values = window.flat::<In, N>().shape() / columns;
        let acc = unit_walk_reduce_microkernel::<In, Acc, N, C>(
            cube,
            input,
            values,
            comptime!(UnitWalkReduceMicrokernel::new(item, per_step)),
        );
        let plane = across_column::<Acc, N, C>(acc, per_column, monoid);
        let lane = UNIT_POS_PLANE as usize;
        let total = C::across_planes(
            plane,
            lane / per_column,
            lane.is_multiple_of(per_column),
            microkernel.planes(),
            columns,
            monoid,
        );
        unit_walk_write_microkernel::<Out, Idx, Acc, N, C>(
            cube,
            total,
            out,
            indices,
            values,
            comptime!(UnitWalkWriteMicrokernel::new(item)),
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
    let acc = unit_strided_reduce_microkernel::<In, Acc, N, C>(
        input,
        unit,
        count,
        first_line,
        comptime!(UnitStridedReduceMicrokernel::new(
            microkernel.item(),
            microkernel.units()
        )),
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
