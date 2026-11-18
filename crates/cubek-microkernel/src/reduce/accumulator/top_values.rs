//! The `K` best values without their indices: a top-k that writes only its values.

use crate::reduce::{ReduceAccumulator, ReduceLayout, ReduceOperation, ReduceWrites};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, Tile, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

#[cube]
impl<Acc: Numeric, N: Size, K: Size> ReduceAccumulator<Acc, N> for TopValues<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, monoid);
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let k = K::value();
        let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(comptime!(planes * k));
        if UNIT_POS_PLANE == 0 {
            stage_plane_list::<Acc, N, K>(&mut values, acc);
        }

        sync_cube();

        let mut total = TopValues::<Acc, N, K>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            #[unroll(unrolled)]
            for slot in 0..k {
                total.insert(values[plane * k + slot], monoid);
            }
        }

        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, _indices: &ComptimeOption<Tile>, part: usize) {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);

        let mut value_lines = values.flat_mut::<Acc, N>();
        #[unroll(unrolled)]
        for slot in 0..k {
            value_lines.write(part * k + slot, acc.values[slot]);
        }
    }

    fn load(values: &Tile, _indices: &ComptimeOption<Tile>, part: usize) -> Self {
        let k = K::value();
        let unrolled = comptime!(k <= TOP_K_UNROLL_BUDGET);

        let value_lines = values.flat::<Acc, N>();
        let mut kept_values = Array::new(k);
        #[unroll(unrolled)]
        for slot in 0..k {
            kept_values[slot] = value_lines.read(part * k + slot);
        }

        TopValues::<Acc, N, K>::new(kept_values)
    }

    /// Where the lanes hold one row, every lane's list is inserted into one one-lane list; one
    /// lane already holds the row's list, best first, and writes it as it is.
    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        _indices: &ComptimeOption<Tile>,
        _count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] _writes: ReduceWrites,
    ) {
        let lanes = N::value();
        let lanes_share_a_row = comptime!(matches!(layout, ReduceLayout::Along) && lanes > 1);
        match lanes_share_a_row {
            true => write_row::<Out, Acc, N, K>(acc, out, operation.monoid()),
            false => write_rows::<Out, Acc, N, K>(acc, out),
        }
    }
}

/// This plane's list, `acc`, staged in its `K` slots of `values`.
#[cube]
fn stage_plane_list<Acc: Numeric, N: Size, K: Size>(
    values: &mut Shared<[Vector<Acc, N>]>,
    acc: TopValues<Acc, N, K>,
) {
    let k = K::value();
    #[unroll]
    for slot in 0..k {
        values[PLANE_POS as usize * k + slot] = acc.values[slot];
    }
}

/// `acc`, whose lanes hold one row, merged into one one-lane list under `monoid` and written to
/// `out`, best first.
#[cube]
fn write_row<Out: Numeric, Acc: Numeric, N: Size, K: Size>(
    acc: TopValues<Acc, N, K>,
    out: &mut Tile,
    #[comptime] monoid: Monoid,
) {
    let k = K::value();
    let unrolled = comptime!(k * k <= TOP_K_UNROLL_BUDGET);
    let mut best = TopValues::<Acc, Const<1>, K>::identity(monoid);
    #[unroll]
    for lane in 0..N::value() {
        #[unroll(unrolled)]
        for slot in 0..k {
            best.insert(Vector::new(acc.values[slot].extract(lane)), monoid);
        }
    }

    let mut lines = out.flat_mut::<Out, Const<1>>();
    #[unroll]
    for slot in 0..k {
        lines.write(slot, Vector::cast_from(best.values[slot]));
    }
}

/// `acc`, each lane a row's list, written to `out`, line `slot` its `slot`-th best.
#[cube]
fn write_rows<Out: Numeric, Acc: Numeric, N: Size, K: Size>(
    acc: TopValues<Acc, N, K>,
    out: &mut Tile,
) {
    let mut lines = out.flat_mut::<Out, N>();
    #[unroll]
    for slot in 0..K::value() {
        lines.write(slot, Vector::cast_from(acc.values[slot]));
    }
}
