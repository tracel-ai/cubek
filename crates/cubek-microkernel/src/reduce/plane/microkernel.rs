use super::base::PlaneShuffleReduceMicrokernel;
use crate::reduce::{
    ReduceAccumulator, ReduceCarrier, accumulate_across, accumulate_strided, row_lanes, row_values,
    values_across, write_across,
};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Region, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One plane's rows, `plane`'s region of `input`, reduced into `out` and `indices` at the same
/// region. Read one line of rows wide, each lane accumulates every `lanes`-th line of the plane's
/// window, consecutive lanes reading consecutive lines, the lanes meet across the plane
/// ([`Carrier::across`](cubecl::tile::Carrier::across)) and the first lane writes the total. Read
/// [`lines_across`](PlaneShuffleReduceMicrokernel::lines_across) lines of rows wide, each column's
/// units take its lines in turns as `plane` deals them, so one load reads whole transactions, and
/// meet within the column.
#[cube]
pub(crate) fn plane_shuffle_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    match microkernel.item().operation.carrier() {
        ReduceCarrier::Values => plane_shuffle_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
            plane,
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::Indexed => plane_shuffle_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
            plane,
            input,
            out,
            indices,
            microkernel,
        ),
        ReduceCarrier::TopK { k } => plane_shuffle_reduce_top_k::<In, Acc, Out, Idx, N>(
            plane,
            input,
            out,
            indices,
            microkernel,
            k,
        ),
        ReduceCarrier::TopValues { k } => plane_shuffle_reduce_top_values::<In, Acc, Out, Idx, N>(
            plane,
            input,
            out,
            indices,
            microkernel,
            k,
        ),
    }
}

#[cube]
fn plane_shuffle_reduce<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    if microkernel.lines_across() == 1 {
        plane_shuffle_reduce_window::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            microkernel,
        );
    } else {
        plane_shuffle_reduce_across::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            microkernel,
        );
    }
}

/// [`plane_shuffle_reduce_microkernel`] read one line of rows wide.
#[cube]
fn plane_shuffle_reduce_window<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    let rows = input.at(plane);
    let mut rows_out = out.at(plane);
    let rows_indices = indices.clone().map(|tile| tile.at(plane));
    let operation = microkernel.item().operation;
    let row_lanes = row_lanes(&rows);
    let count = rows.flat::<In, N>().shape();
    let total = plane_shuffle_reduce_part::<In, Acc, N, C>(&rows, count, microkernel);
    let values = row_values(count, row_lanes);

    if UNIT_POS_PLANE == 0 {
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

/// [`plane_shuffle_reduce_microkernel`] read `lines_across` lines of rows wide. The units of a
/// column hold the low bits of the lane, as `plane`'s units level lays the reduced dimension
/// innermost, so a column meets across those bits alone.
#[cube]
fn plane_shuffle_reduce_across<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) {
    let columns = microkernel.lines_across();
    let per_column = microkernel.lanes() / columns;
    let values = values_across::<In, N>(plane, input, columns);
    let acc =
        accumulate_across::<In, Acc, N, C>(plane, input, values, per_column, microkernel.item());
    let column = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    let total = C::across(acc, column, microkernel.item().operation.monoid());
    write_across::<Out, Idx, Acc, N, C>(plane, total, out, indices, values, microkernel.item());
}

/// The `count` lines of `input` accumulated by the plane into `C`: each lane takes every
/// `lanes`-th line, consecutive lanes reading consecutive lines, and the lanes meet across the
/// plane. Every lane is left with the total.
#[cube]
fn plane_shuffle_reduce_part<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    input: &Tile,
    count: usize,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
) -> C {
    let monoid = microkernel.item().operation.monoid();
    let acc = accumulate_strided::<In, Acc, N, C>(
        input,
        UNIT_POS_PLANE as usize,
        microkernel.lanes(),
        count,
        0,
        microkernel.item(),
    );

    C::across(acc, UnitShare::Plane, monoid)
}

/// [`plane_shuffle_reduce_microkernel`] into a top-k of `k` slots, `K` bound in the
/// function that uses it.
#[cube]
fn plane_shuffle_reduce_top_k<In: Numeric, Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    plane_shuffle_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(
        plane,
        input,
        out,
        indices,
        microkernel,
    )
}

/// [`plane_shuffle_reduce_microkernel`] into a top-k of `k` values without their indices, `K`
/// bound in the function that uses it.
#[cube]
fn plane_shuffle_reduce_top_values<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    plane: &Region,
    input: &Tile,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] microkernel: PlaneShuffleReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    plane_shuffle_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
        plane,
        input,
        out,
        indices,
        microkernel,
    )
}
