//! The `K` best values without their indices: a top-k that writes only its values.

use crate::reduce::{ReduceAccumulator, stage_slot};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, TopValues};
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

    /// As [`TopK`](cubecl::tile::TopK)'s, the lists holding values alone.
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
            stage_plane_list::<Acc, N, K>(
                &mut values,
                stage_slot(PLANE_POS as usize, column, columns),
                acc,
            );
        }

        sync_cube();

        let total = merged_by_first_plane::<Acc, N, K>(&values, column, planes, columns, monoid);

        sync_cube();
        total
    }

    fn value_line(acc: &Self, slot: usize) -> Vector<Acc, N> {
        acc.values[slot]
    }

    fn index_line(_acc: &Self, _slot: usize) -> Vector<u32, N> {
        Vector::new(0u32)
    }

    fn set_line(acc: &mut Self, slot: usize, value: Vector<Acc, N>, _index: Vector<u32, N>) {
        acc.values[slot] = value;
    }

    /// Every lane's list inserted into one one-lane list, best first.
    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self {
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

        let mut kept_values = Array::new(k);
        #[unroll]
        for slot in 0..k {
            kept_values[slot] = Vector::new(best.values[slot].extract(0usize));
        }
        TopValues::<Acc, N, K>::new(kept_values)
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
        let list = stage_slot(plane, column, columns) * k;
        for slot in 0..k {
            total.insert(values[list + slot], monoid);
        }
    }
}
