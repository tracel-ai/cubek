use super::base::HypercubeLastArrivalReduceMicrokernel;
use crate::reduce::cube::{CubePartialReduceMicrokernel, cube_partial_reduce_microkernel};
use crate::reduce::read::{UnitPartialsReduceMicrokernel, unit_partials_reduce_microkernel};
use crate::reduce::{REDUCED, ReduceAccumulator, ReduceCarrier};
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a row its `parts` cubes split: the cube reduces `input`, its window of the
/// part, with the cube's microkernel, each line indexed `first_line` lines further, its place in
/// the whole row; a part the row's end cuts short reads the edge monoid's identity past it. Its first unit
/// stores the partial as partial `part` of the row's `partials` and `partial_indices`, its
/// carrier's lines a slot apart, and the cube counts itself in at the row's `counter` of
/// `arrivals`. The row's last cube to count merges every partial and writes the
/// row to `out`, and to `indices` where both halves are written, as the cube's microkernel writes
/// a row, a mean dividing by `values`, the values the whole row holds.
///
/// `arrivals` holds a counter for every row, or box of rows, its caller counts apart, at zero; the
/// row's last cube sets it back to zero once it has merged, so the caller's counters serve its
/// next launch.
/// The cube meets on `sync_storage` before it counts, so its partial lands first, and again before
/// the last cube reads the others'.
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

/// [`hypercube_last_arrival_reduce_microkernel`] into the carrier `C`.
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
    let cube = microkernel.part();
    cube_partial_reduce_microkernel::<In, Acc, N, C>(
        input,
        partials,
        partial_indices,
        part,
        first_line,
        comptime!(CubePartialReduceMicrokernel::new(cube)),
    );

    // The partial lands at device scope before the cube counts itself in.
    sync_storage();
    let last = counts_in_last(arrivals, counter, parts);
    sync_storage_where(last);

    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));
    let merges = last && UNIT_POS == 0;
    if merges {
        unit_partials_reduce_microkernel::<Out, Idx, Acc, N, C>(
            &*partials,
            partial_indices,
            parts,
            out,
            indices,
            values,
            comptime!(UnitPartialsReduceMicrokernel::new(cube.item(), row_lanes)),
        );
    }
    reset_count_where(arrivals, counter, merges);
}

/// Whether the cube is the last of its row's `parts` to count itself in at `counter` of `arrivals`:
/// its first unit counts it in, and the cube meets so every unit answers the same.
#[cube]
fn counts_in_last(arrivals: &[Atomic<u32>], counter: usize, parts: usize) -> bool {
    let mut arrived = Shared::<[u32]>::new_slice(1usize);
    if UNIT_POS == 0 {
        arrived[0] = arrivals[counter].fetch_add(1u32);
    }

    sync_cube();
    arrived[0] as usize + 1 == parts
}

/// What every other cube of the row stored acquired before it is read, by the cube whose units all
/// hold `last`.
#[cube]
fn sync_storage_where(last: bool) {
    if last {
        sync_storage();
    }
}

/// The counter at `counter` of `arrivals` set back to zero by the unit that `merges` the row, which
/// every other cube of the row counted in before: nothing counts there again in this launch.
#[cube]
fn reset_count_where(arrivals: &[Atomic<u32>], counter: usize, merges: bool) {
    if merges {
        arrivals[counter].store(0u32);
    }
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
