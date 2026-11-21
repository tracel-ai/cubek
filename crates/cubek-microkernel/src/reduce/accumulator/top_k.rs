//! The `K` best values with their indices: the top-k.

use crate::reduce::{ReduceAccumulator, lane_indices};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, TopK};
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
                PLANE_POS as usize * columns + column,
                acc,
            );
        }

        sync_cube();

        let total =
            merged_by_first_plane::<Acc, N, K>(&values, &indices, column, planes, columns, monoid);

        sync_cube();
        total
    }

    fn value_line(acc: &Self, slot: usize) -> Vector<Acc, N> {
        acc.values[slot]
    }

    fn index_line(acc: &Self, slot: usize) -> Vector<u32, N> {
        acc.indices[slot]
    }

    fn set_line(acc: &mut Self, slot: usize, value: Vector<Acc, N>, index: Vector<u32, N>) {
        acc.values[slot] = value;
        acc.indices[slot] = index;
    }

    /// Every lane's list inserted into one one-lane list, best first; one lane already holds the
    /// row's list where the lanes hold one row.
    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self {
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

        let mut kept_values = Array::new(k);
        let mut kept_indices = Array::new(k);
        #[unroll]
        for slot in 0..k {
            kept_values[slot] = Vector::new(best.values[slot].extract(0usize));
            kept_indices[slot] = Vector::new(best.indices[slot].extract(0usize));
        }
        TopK::<Acc, N, K>::new(kept_values, kept_indices)
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
        let list = (plane * columns + column) * k;
        for slot in 0..k {
            total.insert_in_full_order(values[list + slot], indices[list + slot], monoid);
        }
    }
}
