//! The `K` best values with their indices: the top-k.

use crate::reduce::{
    ReduceAccumulator, ReduceLayout, ReduceOperation, ReduceWrites, lane_indices, write_line,
};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, Tile, TopK};
use cubecl::{self as cubecl, cube, prelude::*};

#[cube]
impl<Acc: Numeric, N: Size, K: Size> ReduceAccumulator<Acc, N> for TopK<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, lane_indices::<N>(line_index, layout), monoid);
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);

        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(comptime!(planes * k));
        let mut indices = Shared::<[Vector<u32, N>]>::new_slice(comptime!(planes * k));
        if UNIT_POS_PLANE == 0 {
            stage_plane_list::<Acc, N, K>(&mut values, &mut indices, acc);
        }

        sync_cube();

        let mut total = TopK::<Acc, N, K>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            #[unroll(unrolled)]
            for slot in 0..k {
                total.insert_in_full_order(
                    values[plane * k + slot],
                    indices[plane * k + slot],
                    monoid,
                );
            }
        }

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
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        let lanes = N::value();
        let lanes_share_a_row = comptime!(matches!(layout, ReduceLayout::Along) && lanes > 1);
        match lanes_share_a_row {
            true => write_row::<Out, Idx, Acc, N, K>(acc, out, indices, operation.monoid(), writes),
            false => write_rows::<Out, Idx, Acc, N, K>(acc, out, indices, writes),
        }
    }
}

/// This plane's list, `acc`, staged in its `K` slots of `values` and `indices`.
#[cube]
fn stage_plane_list<Acc: Numeric, N: Size, K: Size>(
    values: &mut Shared<[Vector<Acc, N>]>,
    indices: &mut Shared<[Vector<u32, N>]>,
    acc: TopK<Acc, N, K>,
) {
    let k = K::value();
    #[unroll]
    for slot in 0..k {
        values[PLANE_POS as usize * k + slot] = acc.values[slot];
        indices[PLANE_POS as usize * k + slot] = acc.indices[slot];
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
