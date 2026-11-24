use super::base::PlaneShuffleReduceMicrokernel;
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

/// One plane's rows, `plane`'s region of `input`, reduced into `out` and `indices` at the same
/// region. Read one line of rows wide, each lane accumulates every `lanes`-th line of the plane's
/// window, consecutive lanes reading consecutive lines, the lanes meet across the plane
/// ([`Carrier::across`](cubecl::tile::Carrier::across)) and the first lane writes the total. Read
/// [`lines_across`](PlaneShuffleReduceMicrokernel::lines_across) lines of rows wide, each column's
/// units take its lines in turns as `plane`'s walk gives them, so one load reads whole transactions, and
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

/// [`plane_shuffle_reduce_microkernel`] into the carrier `C`.
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
    match microkernel.lines_across() == 1 {
        true => plane_shuffle_reduce_line::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            microkernel,
        ),
        false => plane_shuffle_reduce_columns::<In, Acc, Out, Idx, N, C>(
            plane,
            input,
            out,
            indices,
            microkernel,
        ),
    }
}

/// [`plane_shuffle_reduce`] one line of rows wide: each lane takes every `lanes`-th line of the
/// plane's window, consecutive lanes reading consecutive lines, and the lanes meet across the plane.
#[cube]
fn plane_shuffle_reduce_line<
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
    let item = microkernel.item();
    let rows = input.at(plane);
    let count = rows.flat::<In, N>().shape();
    let acc = unit_strided_reduce_microkernel::<In, Acc, N, C>(
        &rows,
        UNIT_POS_PLANE as usize,
        count,
        0,
        comptime!(UnitStridedReduceMicrokernel::new(item, microkernel.lanes())),
    );
    let total = C::across(acc, UnitShare::Plane, item.operation.monoid());

    let load = rows.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let mut rows_out = out.at(plane);
    let rows_indices = indices.clone().map(|tile| tile.at(plane));
    if UNIT_POS_PLANE == 0 {
        unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
            total,
            &mut rows_out,
            &rows_indices,
            count * row_lanes,
            comptime!(UnitCarrierWriteMicrokernel::new(item, row_lanes)),
        );
    }
}

/// [`plane_shuffle_reduce`] [`lines_across`](PlaneShuffleReduceMicrokernel::lines_across) lines of
/// rows wide: the units of a column hold the low bits of the lane, as `plane`'s units level lays the
/// reduced dimension innermost, so a column meets across those bits alone.
#[cube]
fn plane_shuffle_reduce_columns<
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
    let item = microkernel.item();
    let columns = microkernel.lines_across();
    let per_column = microkernel.lanes() / columns;
    let window = input.at(plane);
    let values = window.flat::<In, N>().shape() / columns;
    let acc = unit_walk_reduce_microkernel::<In, Acc, N, C>(
        plane,
        input,
        values,
        comptime!(UnitWalkReduceMicrokernel::new(item, per_column)),
    );
    let column = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    let total = C::across(acc, column, item.operation.monoid());
    unit_walk_write_microkernel::<Out, Idx, Acc, N, C>(
        plane,
        total,
        out,
        indices,
        values,
        comptime!(UnitWalkWriteMicrokernel::new(item)),
    );
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
