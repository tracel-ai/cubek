use super::base::HypercubeLastArrivalReduceMicrokernel;
use crate::reduce::write::{UnitCarrierWriteMicrokernel, unit_carrier_write_microkernel};
use crate::reduce::{REDUCED, ReduceAccumulator, ReduceCarrier};
use cubecl::tile::{Indexed, TOP_K_UNROLL_BUDGET, Tile, TopK, TopValues};
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
    let item = cube_part.item();
    let monoid = item.operation.monoid();
    // A partial is its carrier's lines, a slot's apart, the indices beside where it keeps any.
    let slots = item.operation.written_per_row();
    let with_indices = item.operation.carrier().carries_indices();
    let unrolled = comptime!(slots <= TOP_K_UNROLL_BUDGET);

    // Past the row's end the input reads the identity of the operation's edge monoid, which
    // leaves the carrier as it is: every line of the window is read.
    let count = input.flat::<In, N>().shape();
    let load = input.vector_tile();
    let row_lanes = comptime!(load.extent_along(REDUCED));

    let total = cube_part
        .runtime()
        .reduce_part::<In, Acc, N, C>(input, first_line, count);

    // The cube's first unit stores its partial as partial `part` of the row.
    if UNIT_POS == 0 {
        let mut value_lines = partials.flat_mut::<Acc, N>();
        #[unroll(unrolled)]
        for slot in 0..slots {
            value_lines.write(part * slots + slot, C::value_line(&total, slot));
        }
        if with_indices {
            let mut index_tile = partial_indices.clone().unwrap();
            let mut index_lines = index_tile.flat_mut::<u32, N>();
            #[unroll(unrolled)]
            for slot in 0..slots {
                index_lines.write(part * slots + slot, C::index_line(&total, slot));
            }
        }
    }

    // The partial lands at device scope before the cube counts itself in.
    sync_storage();

    let arrived = count_in(arrivals, counter);

    // Uniform across the cube: only the row's last cube merges.
    if arrived as usize + 1 == parts {
        // What every other cube of the row stored is acquired before it is read.
        sync_storage();

        if UNIT_POS == 0 {
            let value_lines = partials.flat::<Acc, N>();
            let mut merged = C::identity(monoid);
            for at in 0..parts {
                let mut stored = C::identity(monoid);
                #[unroll(unrolled)]
                for slot in 0..slots {
                    let line = at * slots + slot;
                    let index = match with_indices {
                        true => {
                            let index_tile = partial_indices.clone().unwrap();
                            index_tile.flat::<u32, N>().read(line)
                        }
                        false => Vector::new(0u32),
                    };
                    C::set_line(&mut stored, slot, value_lines.read(line), index);
                }
                merged = C::combine(merged, stored, monoid);
            }
            unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
                merged,
                out,
                indices,
                values,
                comptime!(UnitCarrierWriteMicrokernel::new(item, row_lanes)),
            );
        }
        reset_count(arrivals, counter);
    }
}

/// How many of its row's cubes had counted themselves in at `counter` of `arrivals` before this
/// one: its first unit counts it in, and the cube meets so every unit reads the count.
#[cube]
fn count_in(arrivals: &[Atomic<u32>], counter: usize) -> u32 {
    let mut arrived = Shared::<[u32]>::new_slice(1usize);
    if UNIT_POS == 0 {
        arrived[0] = arrivals[counter].fetch_add(1u32);
    }

    sync_cube();
    arrived[0]
}

/// The counter at `counter` of `arrivals` set back to zero by the row's last cube, which every
/// other cube of the row counted in before: nothing counts there again in this launch.
#[cube]
fn reset_count(arrivals: &[Atomic<u32>], counter: usize) {
    if UNIT_POS == 0 {
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
