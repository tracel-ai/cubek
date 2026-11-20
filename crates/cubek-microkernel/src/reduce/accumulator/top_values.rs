//! The `K` best values without their indices: a top-k that writes only its values.

use crate::reduce::{ReduceAccumulator, ReduceOperation, ReduceWrites};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, Tile, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

#[cube]
impl<Acc: Numeric, N: Size, K: Size> ReduceAccumulator<Acc, N> for TopValues<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _row_lanes: usize,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, monoid);
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
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(comptime!(planes * columns * k));
        if stages {
            stage_plane_list::<Acc, N, K>(&mut values, PLANE_POS as usize * columns + column, acc);
        }

        sync_cube();

        let total = merged_by_first_plane::<Acc, N, K>(&values, column, planes, columns, monoid);

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
        #[comptime] row_lanes: usize,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        comptime!(assert!(
            writes == ReduceWrites::Values,
            "ReduceAccumulator: a top-k of values alone holds no index to write"
        ));
        match row_lanes > 1 {
            true => write_row::<Out, Acc, N, K>(acc, out, operation.monoid()),
            false => write_rows::<Out, Acc, N, K>(acc, out),
        }
    }
}

/// A plane's list, `acc`, staged as list `list` of `values`, `K` slots each.
#[cube]
fn stage_plane_list<Acc: Numeric, N: Size, K: Size>(
    values: &mut Shared<[Vector<Acc, N>]>,
    list: usize,
    acc: TopValues<Acc, N, K>,
) {
    let k = K::value();
    #[unroll]
    for slot in 0..k {
        values[list * k + slot] = acc.values[slot];
    }
}

/// The lists of `column` the `planes` planes staged in `values`, `columns` lists a plane, merged
/// by the cube's first plane; every other unit is left with the identity.
#[cube]
fn merged_by_first_plane<Acc: Numeric, N: Size, K: Size>(
    values: &Shared<[Vector<Acc, N>]>,
    column: usize,
    #[comptime] planes: usize,
    #[comptime] columns: usize,
    #[comptime] monoid: Monoid,
) -> TopValues<Acc, N, K> {
    let mut total = TopValues::<Acc, N, K>::identity(monoid);
    if PLANE_POS == 0 {
        insert_plane_lists::<Acc, N, K>(&mut total, values, column, planes, columns, monoid);
    }
    total
}

/// The lists of `column` the `planes` planes staged in `values`, `columns` lists a plane,
/// inserted into `total`.
#[cube]
fn insert_plane_lists<Acc: Numeric, N: Size, K: Size>(
    total: &mut TopValues<Acc, N, K>,
    values: &Shared<[Vector<Acc, N>]>,
    column: usize,
    #[comptime] planes: usize,
    #[comptime] columns: usize,
    #[comptime] monoid: Monoid,
) {
    let k = K::value();
    for plane in 0..planes {
        let list = (plane * columns + column) * k;
        for slot in 0..k {
            total.insert(values[list + slot], monoid);
        }
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
