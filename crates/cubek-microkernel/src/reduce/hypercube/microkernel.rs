use super::base::HypercubeReduceMicrokernel;
use crate::reduce::{ReduceAccumulator, ReduceCarrier, ReduceLayout, read_mapped};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Tile, TopK, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// One cube's part of a row its `parts` cubes split: the cube reduces `input`, its window of the
/// part, as the cube's microkernel reduces a row, each line indexed `first_line` lines further, its
/// place in the whole row; a part the row's end cuts short reads no line past it. Its first unit
/// stores the partial as partial `part` of the row's `partials` and `partial_indices`
/// ([`ReduceAccumulator::store`]), and the cube counts itself in at the row's `counter` of
/// `arrivals`. The row's last cube to count merges every partial and writes the
/// row to `out`, and to `indices` where both halves are written, as the cube's microkernel writes
/// a row, a mean dividing by `values`, the values the whole row holds.
///
/// `arrivals` holds a counter for every row, at zero; the row's last cube leaves it at zero again.
/// The cube meets on `sync_storage` before it counts, so its partial lands first, and again before
/// the last cube reads the others'.
///
/// Its operation picks what the row accumulates into ([`ReduceCarrier`]).
#[cube]
#[allow(clippy::too_many_arguments)]
pub fn hypercube_reduce_microkernel<
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
    #[comptime] microkernel: HypercubeReduceMicrokernel,
) {
    match comptime!(microkernel.item.operation.carrier()) {
        ReduceCarrier::Values => hypercube_reduce::<In, Acc, Out, Idx, N, Vector<Acc, N>>(
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
        ),
        ReduceCarrier::Indexed => hypercube_reduce::<In, Acc, Out, Idx, N, Indexed<Acc, N>>(
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
        ),
        ReduceCarrier::TopK { k } => {
            let size!(K) = k;
            hypercube_reduce::<In, Acc, Out, Idx, N, TopK<Acc, N, K>>(
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
            hypercube_reduce::<In, Acc, Out, Idx, N, TopValues<Acc, N, K>>(
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
#[allow(clippy::too_many_arguments)]
fn hypercube_reduce<
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
    #[comptime] microkernel: HypercubeReduceMicrokernel,
) {
    let operation = comptime!(microkernel.item.operation);
    let monoid = comptime!(operation.monoid());
    let layout = comptime!(microkernel.item.layout);
    let units = comptime!(microkernel.lanes * microkernel.planes);
    let unit = PLANE_POS as usize * comptime!(microkernel.lanes) + UNIT_POS_PLANE as usize;
    let lines = input.flat::<In, N>();
    // The tile reads zero past the row's end, which no carrier holds as its identity.
    let row_lines = match comptime!(layout) {
        ReduceLayout::Along => values / N::value(),
        ReduceLayout::Across => values,
    };
    let count = lines.shape().min(row_lines - first_line);
    let mut acc = C::identity(monoid);
    for step in 0..count.div_ceil(units) {
        let line = unit + step * units;
        if line < count {
            let read = read_mapped::<Acc, N>(
                Vector::<Acc, N>::cast_from(lines.read(line)),
                comptime!(operation.read_map()),
            );
            C::accumulate(&mut acc, read, first_line + line, layout, monoid);
        }
    }
    let plane = C::across(acc, UnitShare::Plane, monoid);
    let total = C::across_planes(plane, comptime!(microkernel.planes), monoid);
    if unit == 0 {
        C::store(total, partials, partial_indices, part);
    }

    // The partial lands at device scope before the cube counts itself in.
    sync_storage();

    let mut arrived = Shared::<[u32]>::new_slice(1usize);
    if unit == 0 {
        arrived[0] = arrivals[counter].fetch_add(1u32);
    }

    sync_cube();

    // Uniform across the cube: only the row's last cube merges.
    if arrived[0] as usize + 1 == parts {
        // What every other cube of the row stored is acquired before it is read.
        sync_storage();

        if unit == 0 {
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
                comptime!(microkernel.item.writes()),
            );
            arrivals[counter].store(0u32);
        }
    }
}
