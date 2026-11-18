use super::base::HypercubeLastArrivalReduceMicrokernel;
use crate::reduce::{ReduceAccumulator, ReduceCarrier};
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a row its `parts` cubes split: the cube reduces `input`, its window of the
/// part, with the cube's microkernel, each line indexed `first_line` lines further, its place in
/// the whole row; a part the row's end cuts short reads the edge monoid's identity past it. Its first unit
/// stores the partial as partial `part` of the row's `partials` and `partial_indices`
/// ([`ReduceAccumulator::store`]), and the cube counts itself in at the row's `counter` of
/// `arrivals`. The row's last cube to count merges every partial and writes the
/// row to `out`, and to `indices` where both halves are written, as the cube's microkernel writes
/// a row, a mean dividing by `values`, the values the whole row holds.
///
/// `arrivals` holds a counter for every row, at zero, which serves this launch alone.
/// The cube meets on `sync_storage` before it counts, so its partial lands first, and again before
/// the last cube reads the others'.
///
/// Its operation picks what the row accumulates into ([`ReduceCarrier`]).
#[cube]
pub(crate) fn hypercube_last_arrival_reduce_microkernel<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    arrivals: &[Atomic<u32>],
    counter: usize,
    part: usize,
    parts: usize,
    first_line: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
) {
    match microkernel.part().item().operation.carrier() {
        ReduceCarrier::Values => {
            hypercube_last_arrival_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
                input,
                out,
                indices,
                partials,
                partial_indices,
                arrivals,
                counter,
                part,
                parts,
                first_line,
                values,
                microkernel,
            )
        }
        ReduceCarrier::Indexed => {
            hypercube_last_arrival_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
                input,
                out,
                indices,
                partials,
                partial_indices,
                arrivals,
                counter,
                part,
                parts,
                first_line,
                values,
                microkernel,
            )
        }
        ReduceCarrier::TopK { k } => hypercube_last_arrival_reduce_top_k::<In, Acc, Out, Idx, N>(
            input,
            out,
            indices,
            partials,
            partial_indices,
            arrivals,
            counter,
            part,
            parts,
            first_line,
            values,
            microkernel,
            k,
        ),
        ReduceCarrier::TopValues { k } => {
            hypercube_last_arrival_reduce_top_values::<In, Acc, Out, Idx, N>(
                input,
                out,
                indices,
                partials,
                partial_indices,
                arrivals,
                counter,
                part,
                parts,
                first_line,
                values,
                microkernel,
                k,
            )
        }
    }
}

#[cube]
fn hypercube_last_arrival_reduce<
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
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    arrivals: &[Atomic<u32>],
    counter: usize,
    part: usize,
    parts: usize,
    first_line: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
) {
    let cube_part = microkernel.part();

    // Past the row's end the input reads the identity of the operation's edge monoid, which
    // leaves the carrier as it is: every line of the window is read.
    let count = input.flat::<In, N>().shape();

    let total = cube_part
        .runtime()
        .reduce_part::<In, Acc, N, C>(input, first_line, count);
    store_partial::<Acc, N, C>(total, partials, partial_indices, part);

    // The partial lands at device scope before the cube counts itself in.
    sync_storage();

    let arrived = count_in(arrivals, counter);

    // Uniform across the cube: only the row's last cube merges.
    if arrived as usize + 1 == parts {
        merge_parts::<Acc, Out, Idx, N, C>(
            out,
            indices,
            partials,
            partial_indices,
            parts,
            values,
            microkernel,
        );
    }
}

/// The cube's partial, `total`, stored by its first unit as partial `part` of its row.
#[cube]
fn store_partial<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    total: C,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    part: usize,
) {
    if PLANE_POS == 0 && UNIT_POS_PLANE == 0 {
        C::store(total, partials, partial_indices, part);
    }
}

/// How many of its row's cubes had counted themselves in at `counter` of `arrivals` before this
/// one: its first unit counts it in, and the cube meets so every unit reads the count.
#[cube]
fn count_in(arrivals: &[Atomic<u32>], counter: usize) -> u32 {
    let mut arrived = Shared::<[u32]>::new_slice(1usize);
    if PLANE_POS == 0 && UNIT_POS_PLANE == 0 {
        arrived[0] = arrivals[counter].fetch_add(1u32);
    }

    sync_cube();
    arrived[0]
}

/// The row's `parts` partials, once every cube of the row stored its own, merged by the last
/// cube's first unit and written as its cube microkernel writes a row.
#[cube]
fn merge_parts<Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    parts: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
) {
    // What every other cube of the row stored is acquired before it is read.
    sync_storage();

    if PLANE_POS == 0 && UNIT_POS_PLANE == 0 {
        write_merged::<Acc, Out, Idx, N, C>(
            out,
            indices,
            partials,
            partial_indices,
            parts,
            values,
            microkernel,
        );
    }
}

/// The row's `parts` partials combined, and the total written to `out` and `indices`.
#[cube]
fn write_merged<Acc: Numeric, Out: Numeric, Idx: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    parts: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
) {
    let item = microkernel.part().item();
    let monoid = item.operation.monoid();
    let stored = partials.clone();
    let mut merged = C::identity(monoid);
    for at in 0..parts {
        merged = C::combine(merged, C::load(&stored, partial_indices, at), monoid);
    }
    C::write::<Out, Idx>(
        merged,
        out,
        indices,
        values,
        item.layout,
        item.operation,
        item.writes(),
    );
}

/// [`hypercube_last_arrival_reduce`] into a top-k of `k` slots, `K` bound in the function that uses
/// it.
#[cube]
fn hypercube_last_arrival_reduce_top_k<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    arrivals: &[Atomic<u32>],
    counter: usize,
    part: usize,
    parts: usize,
    first_line: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    hypercube_last_arrival_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(
        input,
        out,
        indices,
        partials,
        partial_indices,
        arrivals,
        counter,
        part,
        parts,
        first_line,
        values,
        microkernel,
    )
}

/// [`hypercube_last_arrival_reduce`] into a top-k of `k` values without their indices, `K` bound in
/// the function that uses it.
#[cube]
fn hypercube_last_arrival_reduce_top_values<
    In: Numeric,
    Acc: Numeric,
    Out: Numeric,
    Idx: Numeric,
    N: Size,
>(
    input: &Tile,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    partials: &mut Tile,
    partial_indices: &ComptimeOption<Tile>,
    arrivals: &[Atomic<u32>],
    counter: usize,
    part: usize,
    parts: usize,
    first_line: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
    #[comptime] k: usize,
) {
    let size!(K) = k;
    hypercube_last_arrival_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
        input,
        out,
        indices,
        partials,
        partial_indices,
        arrivals,
        counter,
        part,
        parts,
        first_line,
        values,
        microkernel,
    )
}
