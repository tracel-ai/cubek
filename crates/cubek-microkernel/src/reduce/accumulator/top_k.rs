//! The `K` best values with their indices: the top-k.

use crate::reduce::{Accumulable, ReduceCarrierFamily, lane_indices};
use cubecl::tile::{Carrier, Monoid, TOP_K_UNROLL_BUDGET, TopK};
use cubecl::{self as cubecl, cube, prelude::*};

/// [`ReduceCarrier::TopK`](crate::reduce::ReduceCarrier::TopK): its carrier is [`TopK`].
pub struct TopKReduceCarrierFamily;

impl ReduceCarrierFamily for TopKReduceCarrierFamily {
    type Carrier<Acc: Numeric, N: Size, K: Size> = TopK<Acc, N, K>;
}

#[cube]
impl<Acc: Numeric, N: Size, K: Size> Accumulable<Acc, N> for TopK<Acc, N, K> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::TestRuntime;

    /// Slot `slot`'s value of the list `first`, then `second` a line later, are inserted into, each
    /// lane its own row.
    #[cube(harness)]
    fn value_kept<N: Size>(
        first: Vector<f32, N>,
        second: Vector<f32, N>,
        #[comptime] slot: usize,
    ) -> Vector<f32, N> {
        let a_row_a_lane = comptime!(1usize);
        let mut acc = TopK::<f32, N, Const<2>>::identity(Monoid::Max);
        TopK::<f32, N, Const<2>>::accumulate(&mut acc, first, 0, a_row_a_lane, Monoid::Max);
        TopK::<f32, N, Const<2>>::accumulate(&mut acc, second, 1, a_row_a_lane, Monoid::Max);
        TopK::<f32, N, Const<2>>::value_line(&acc, slot)
    }

    /// Slot `slot`'s value once slot 0 is set to `first` and slot 1 to `second`, the lanes one row.
    #[cube(harness)]
    fn value_of_one_row<N: Size>(
        first: Vector<f32, N>,
        second: Vector<f32, N>,
        #[comptime] slot: usize,
    ) -> Vector<f32, N> {
        let index = Vector::new(0u32);
        let mut acc = TopK::<f32, N, Const<2>>::identity(Monoid::Max);
        TopK::<f32, N, Const<2>>::set_line(&mut acc, 0, first, index);
        TopK::<f32, N, Const<2>>::set_line(&mut acc, 1, second, index);
        let row = TopK::<f32, N, Const<2>>::one_row(acc, Monoid::Max);
        TopK::<f32, N, Const<2>>::value_line(&row, slot)
    }

    #[test]
    fn the_best_line_is_kept_first() {
        let value = value_kept::harness::<TestRuntime, Const<2>>(&[1.0, 5.0], &[4.0, 2.0], 0);
        assert_eq!(value, [4.0, 5.0]);
    }

    #[test]
    fn the_next_best_is_kept_second() {
        let value = value_kept::harness::<TestRuntime, Const<2>>(&[1.0, 5.0], &[4.0, 2.0], 1);
        assert_eq!(value, [1.0, 2.0]);
    }

    #[test]
    fn one_row_ranks_every_lanes_list_in_every_lane() {
        let value = value_of_one_row::harness::<TestRuntime, Const<2>>(&[9.0, 7.0], &[3.0, 1.0], 1);
        assert_eq!(value, [7.0, 7.0]);
    }

    /// Slot `slot`'s index of the list `first`, then `second` a line later, are inserted into.
    #[cube(harness)]
    fn index_kept<N: Size>(
        first: Vector<f32, N>,
        second: Vector<f32, N>,
        #[comptime] slot: usize,
    ) -> Vector<u32, N> {
        let a_row_a_lane = comptime!(1usize);
        let mut acc = TopK::<f32, N, Const<2>>::identity(Monoid::Max);
        TopK::<f32, N, Const<2>>::accumulate(&mut acc, first, 0, a_row_a_lane, Monoid::Max);
        TopK::<f32, N, Const<2>>::accumulate(&mut acc, second, 1, a_row_a_lane, Monoid::Max);
        TopK::<f32, N, Const<2>>::index_line(&acc, slot)
    }

    #[test]
    fn each_values_index_is_kept_beside_it() {
        let index = index_kept::harness::<TestRuntime, Const<2>>(&[1.0, 5.0], &[4.0, 2.0], 0);
        assert_eq!(index, [1, 0]);
    }
}
