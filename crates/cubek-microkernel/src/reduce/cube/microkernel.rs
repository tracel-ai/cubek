use super::base::CubeStagedReduceMicrokernel;
use crate::reduce::read::{
    UnitStridedReduceMicrokernel, UnitWalkReduceMicrokernel, unit_strided_reduce_microkernel,
    unit_walk_reduce_microkernel,
};
use crate::reduce::write::{
    UnitCarrierWriteMicrokernel, UnitWalkWriteMicrokernel, unit_carrier_write_microkernel,
    unit_walk_write_microkernel,
};
use crate::reduce::{REDUCED, ReduceAccumulator, ReduceCarrier};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Region, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's rows, `cube`'s region of `input`, reduced into `out` and `indices` at the same
/// region. Read one line of rows wide, each of its units accumulates every `units`-th line of the
/// cube's window, consecutive units reading consecutive lines; each plane's lanes meet in the
/// plane's instructions, the planes meet in shared memory
/// ([`across_planes`]), and the cube's first unit writes the total. Read
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

        if UNIT_POS == 0 {
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
        let column = comptime!(UnitShare::Group {
            unit_bits: per_column - 1
        });
        let plane = C::across(acc, column, monoid);
        let lane = UNIT_POS_PLANE as usize;
        let total = across_planes::<Acc, N, C>(
            plane,
            lane / per_column,
            lane.is_multiple_of(per_column),
            columns,
            microkernel,
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
/// shared memory ([`across_planes`]). The cube's first unit is left with the
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
    across_planes::<Acc, N, C>(plane, 0, UNIT_POS_PLANE == 0, one_column, microkernel)
}

/// The partials of the cube's planes, each already met across its own lanes, met in shared
/// memory, one total for each of the `columns` lines of rows its units lie across: the unit of
/// each plane and column where `stages` is set stages its partial, `acc`, in that column's slot of
/// its plane, the cube meets, the first plane's units combine their column's slots, and the cube
/// meets again so the slots may be staged anew. Only the first plane's units are left with their
/// column's total; the others hold the identity.
#[cube]
fn across_planes<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    acc: C,
    column: usize,
    stages: bool,
    #[comptime] columns: usize,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) -> C {
    let operation = microkernel.item().operation;
    // A partial is its carrier's lines, a slot each, the indices beside where it keeps any.
    let lines = microkernel.planes() * columns * operation.written_per_row();
    let index_lines = comptime!(match operation.carrier().carries_indices() {
        true => lines,
        false => 1,
    });
    let mut values = Shared::<[Vector<Acc, N>]>::new_slice(lines);
    let mut indices = Shared::<[Vector<u32, N>]>::new_slice(index_lines);
    if stages {
        stage_partial::<Acc, N, C>(
            &mut values,
            &mut indices,
            PLANE_POS as usize * columns + column,
            acc,
            microkernel,
        );
    }

    sync_cube();

    let total = merged_by_first_plane::<Acc, N, C>(&values, &indices, column, columns, microkernel);

    sync_cube();
    total
}

/// `acc` staged as partial `partial` of `values` and `indices`, its slots' lines side by side.
#[cube]
fn stage_partial<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &mut Shared<[Vector<Acc, N>]>,
    indices: &mut Shared<[Vector<u32, N>]>,
    partial: usize,
    acc: C,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) {
    let operation = microkernel.item().operation;
    let slots = operation.written_per_row();
    let with_indices = operation.carrier().carries_indices();
    #[unroll]
    for slot in 0..slots {
        values[partial * slots + slot] = C::value_line(&acc, slot);
        if with_indices {
            indices[partial * slots + slot] = C::index_line(&acc, slot);
        }
    }
}

/// The partials of `column` the cube's planes staged in `values` and `indices`, `columns` a
/// plane, combined by the cube's first plane; every other unit is left with the identity.
#[cube]
fn merged_by_first_plane<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] columns: usize,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) -> C {
    let mut total = C::identity(microkernel.item().operation.monoid());
    if PLANE_POS == 0 {
        total = merged_partials::<Acc, N, C>(values, indices, column, columns, microkernel);
    }
    total
}

/// The partials of `column` the cube's planes staged in `values` and `indices`, `columns` a
/// plane, combined. A carrier of one slot unrolls its loop over the planes; a top-k's stays a
/// loop, as each of its partials is itself `k` lines.
#[cube]
fn merged_partials<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] columns: usize,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) -> C {
    let operation = microkernel.item().operation;
    let monoid = operation.monoid();
    let one_slot = operation.written_per_row() == 1;
    let mut total = C::identity(monoid);
    #[unroll(one_slot)]
    for plane in 0..microkernel.planes() {
        let partial =
            staged_partial::<Acc, N, C>(values, indices, plane * columns + column, microkernel);
        total = C::combine(total, partial, monoid);
    }
    total
}

/// Partial `partial` of `values` and `indices` as [`stage_partial`] staged it.
#[cube]
fn staged_partial<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    partial: usize,
    #[comptime] microkernel: CubeStagedReduceMicrokernel,
) -> C {
    let operation = microkernel.item().operation;
    let slots = operation.written_per_row();
    let with_indices = operation.carrier().carries_indices();
    let mut staged = C::identity(operation.monoid());
    #[unroll]
    for slot in 0..slots {
        let index = match with_indices {
            true => indices[partial * slots + slot],
            false => Vector::new(0u32),
        };
        C::set_line(&mut staged, slot, values[partial * slots + slot], index);
    }
    staged
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
