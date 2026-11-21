//! The `K` best values with their indices: the top-k.

use crate::reduce::{
    ReduceAccumulator, ReduceOperation, ReduceWrites, lane_indices, stage_slot, write_line,
};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, Tile, TopK};
use cubecl::{self as cubecl, cube, prelude::*};

#[cube]
impl<Acc: Numeric, N: Size, K: Size> ReduceAccumulator<Acc, N> for TopK<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] row_lanes: usize,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, lane_indices::<N>(line_index, row_lanes), monoid);
    }

    /// Only the first plane, which holds the cube's first unit, merges the planes' lists, and
    /// its loop over them stays a loop: only each value's walk unrolls
    /// ([`TOP_K_UNROLL_BUDGET`]).
    fn across_planes(
        acc: Self,
        column: usize,
        stages: bool,
        #[comptime] planes: usize,
        #[comptime] columns: usize,
        #[comptime] monoid: Monoid,
    ) -> Self {
        let k = K::value();
        let slots = comptime!(planes * columns * k);
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(slots);
        let mut indices = Shared::<[Vector<u32, N>]>::new_slice(slots);
        if stages {
            stage_plane_list::<Acc, N, K>(
                &mut values,
                &mut indices,
                stage_slot(PLANE_POS as usize, column, columns),
                acc,
            );
        }

        sync_cube();

        let total =
            merged_by_first_plane::<Acc, N, K>(&values, &indices, column, planes, columns, monoid);

        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, indices: &ComptimeOption<Tile>, part: usize) {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);

        let mut value_lines = values.flat_mut::<Acc, N>();
        let mut index_tile = indices.clone().unwrap();
        let mut index_lines = index_tile.flat_mut::<u32, N>();
        #[unroll(unrolled)]
        for slot in 0..k {
            value_lines.write(part * k + slot, acc.values[slot]);
            index_lines.write(part * k + slot, acc.indices[slot]);
        }
    }

    fn load(values: &Tile, indices: &ComptimeOption<Tile>, part: usize) -> Self {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);

        let value_lines = values.flat::<Acc, N>();
        let index_tile = indices.clone().unwrap();
        let index_lines = index_tile.flat::<u32, N>();
        let mut kept_values = Array::new(k);
        let mut kept_indices = Array::new(k);
        #[unroll(unrolled)]
        for slot in 0..k {
            kept_values[slot] = value_lines.read(part * k + slot);
            kept_indices[slot] = index_lines.read(part * k + slot);
        }

        TopK::<Acc, N, K>::new(kept_values, kept_indices)
    }

    /// Where the lanes hold one row, every lane's list is inserted into one one-lane list; one
    /// lane already holds the row's list, best first, and writes it as it is.
    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        _count: usize,
        #[comptime] row_lanes: usize,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        match row_lanes > 1 {
            true => write_row::<Out, Idx, Acc, N, K>(acc, out, indices, operation.monoid(), writes),
            false => write_rows::<Out, Idx, Acc, N, K>(acc, out, indices, writes),
        }
    }
}

/// A plane's list, `acc`, staged as list `list` of `values` and `indices`, `K` slots each.
#[cube]
fn stage_plane_list<Acc: Numeric, N: Size, K: Size>(
    values: &mut Shared<[Vector<Acc, N>]>,
    indices: &mut Shared<[Vector<u32, N>]>,
    list: usize,
    acc: TopK<Acc, N, K>,
) {
    let k = K::value();
    #[unroll]
    for slot in 0..k {
        values[list * k + slot] = acc.values[slot];
        indices[list * k + slot] = acc.indices[slot];
    }
}

/// The lists of `column` the `planes` planes staged in `values` and `indices`, `columns` lists a
/// plane, merged by the cube's first plane; every other unit is left with the identity.
#[cube]
fn merged_by_first_plane<Acc: Numeric, N: Size, K: Size>(
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] planes: usize,
    #[comptime] columns: usize,
    #[comptime] monoid: Monoid,
) -> TopK<Acc, N, K> {
    let mut total = TopK::<Acc, N, K>::identity(monoid);
    if PLANE_POS == 0 {
        insert_plane_lists::<Acc, N, K>(
            &mut total, values, indices, column, planes, columns, monoid,
        );
    }
    total
}

/// The lists of `column` the `planes` planes staged in `values` and `indices`, `columns` lists a
/// plane, inserted into `total`.
#[cube]
fn insert_plane_lists<Acc: Numeric, N: Size, K: Size>(
    total: &mut TopK<Acc, N, K>,
    values: &Shared<[Vector<Acc, N>]>,
    indices: &Shared<[Vector<u32, N>]>,
    column: usize,
    #[comptime] planes: usize,
    #[comptime] columns: usize,
    #[comptime] monoid: Monoid,
) {
    let k = K::value();
    for plane in 0..planes {
        let list = stage_slot(plane, column, columns) * k;
        for slot in 0..k {
            total.insert_in_full_order(values[list + slot], indices[list + slot], monoid);
        }
    }
}

/// `acc`, whose lanes hold one row, merged into one one-lane list under `monoid` and written as
/// `writes` says, best first.
#[cube]
fn write_row<Out: Numeric, Idx: Numeric, Acc: Numeric, N: Size, K: Size>(
    acc: TopK<Acc, N, K>,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] monoid: Monoid,
    #[comptime] writes: ReduceWrites,
) {
    let k = K::value();
    let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
    let mut best = TopK::<Acc, Const<1>, K>::identity(monoid);
    #[unroll]
    for lane in 0..N::value() {
        #[unroll(unrolled)]
        for slot in 0..k {
            best.insert_in_full_order(
                Vector::new(acc.values[slot].extract(lane)),
                Vector::new(acc.indices[slot].extract(lane)),
                monoid,
            );
        }
    }

    #[unroll]
    for slot in 0..k {
        write_line::<Out, Idx, Acc, Const<1>>(
            out,
            indices,
            slot,
            best.values[slot],
            best.indices[slot],
            writes,
        );
    }
}

/// `acc`, each lane a row's list, written as `writes` says, line `slot` its `slot`-th best.
#[cube]
fn write_rows<Out: Numeric, Idx: Numeric, Acc: Numeric, N: Size, K: Size>(
    acc: TopK<Acc, N, K>,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] writes: ReduceWrites,
) {
    #[unroll]
    for slot in 0..K::value() {
        write_line::<Out, Idx, Acc, N>(
            out,
            indices,
            slot,
            acc.values[slot],
            acc.indices[slot],
            writes,
        );
    }
}
