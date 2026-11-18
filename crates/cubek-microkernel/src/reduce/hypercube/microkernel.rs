use super::base::HypercubeLastArrivalReduceMicrokernel;
use crate::reduce::cube::cube_staged_reduce_part;
use crate::reduce::{ReduceAccumulator, ReduceCarrier, ReduceLayout};
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a row its `parts` cubes split: the cube reduces `input`, its window of the
/// part, with the cube's microkernel, each line indexed `first_line` lines further, its place in
/// the whole row; a part the row's end cuts short reads no line past it. Its first unit
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
    indices: &mut Tile,
    partials: &mut Tile,
    partial_indices: &mut Tile,
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
        ReduceCarrier::TopK { k } => {
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
        ReduceCarrier::TopValues { k } => {
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
    indices: &mut Tile,
    partials: &mut Tile,
    partial_indices: &mut Tile,
    arrivals: &[Atomic<u32>],
    counter: usize,
    part: usize,
    parts: usize,
    first_line: usize,
    values: usize,
    #[comptime] microkernel: HypercubeLastArrivalReduceMicrokernel,
) {
    let cube_part = microkernel.part();
    let operation = cube_part.item().operation;
    let monoid = operation.monoid();
    let layout = cube_part.item().layout;
    let first_unit = PLANE_POS == 0 && UNIT_POS_PLANE == 0;

    // The tile reads zero past the row's end, which no carrier holds as its identity.
    let row_lines = match layout {
        ReduceLayout::Along => values / N::value(),
        ReduceLayout::Across => values,
    };
    let count = input.flat::<In, N>().shape().min(row_lines - first_line);

    let total = cube_staged_reduce_part::<In, Acc, N, C>(input, first_line, count, cube_part);

    if first_unit {
        C::store(total, partials, partial_indices, part);
    }

    // The partial lands at device scope before the cube counts itself in.
    sync_storage();

    let mut arrived = Shared::<[u32]>::new_slice(1usize);
    if first_unit {
        arrived[0] = arrivals[counter].fetch_add(1u32);
    }

    sync_cube();

    // Uniform across the cube: only the row's last cube merges.
    if arrived[0] as usize + 1 == parts {
        // What every other cube of the row stored is acquired before it is read.
        sync_storage();

        if first_unit {
            let (stored, stored_indices) = (partials.clone(), partial_indices.clone());
            let mut merged = C::identity(monoid);
            for at in 0..parts {
                merged = C::combine(merged, C::load(&stored, &stored_indices, at), monoid);
            }
            C::write::<Out, Idx>(
                merged,
                out,
                indices,
                values,
                layout,
                operation,
                cube_part.item().writes(),
            );
        }
    }
}
