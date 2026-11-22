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
