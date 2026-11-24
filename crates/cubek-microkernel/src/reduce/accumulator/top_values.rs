//! The `K` best values without their indices: a top-k that writes only its values.

use crate::reduce::{Accumulable, ReduceCarrierFamily, unrolls_a_walk_of_count};
use cubecl::tile::{Carrier, Monoid, TopValues};
use cubecl::{self as cubecl, cube, prelude::*};

/// [`ReduceCarrier::TopValues`](crate::reduce::ReduceCarrier::TopValues): its carrier is [`TopValues`].
pub struct TopValuesReduceCarrierFamily;

impl ReduceCarrierFamily for TopValuesReduceCarrierFamily {
    type Carrier<Acc: Numeric, N: Size, K: Size> = TopValues<Acc, N, K>;
}

#[cube]
impl<Acc: Numeric, N: Size, K: Size> Accumulable<Acc, N> for TopValues<Acc, N, K> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _row_lanes: usize,
        #[comptime] monoid: Monoid,
    ) {
        acc.insert(line, monoid);
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
        let unrolled = comptime!(unrolls_a_walk_of_count(k));
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::TestRuntime;

    #[cube(harness)]
    fn value_kept<N: Size>(
        first: Vector<f32, N>,
        second: Vector<f32, N>,
        #[comptime] slot: usize,
    ) -> Vector<f32, N> {
        let a_row_a_lane = comptime!(1usize);
        let mut acc = TopValues::<f32, N, Const<2>>::identity(Monoid::Max);
        TopValues::<f32, N, Const<2>>::accumulate(&mut acc, first, 0, a_row_a_lane, Monoid::Max);
        TopValues::<f32, N, Const<2>>::accumulate(&mut acc, second, 1, a_row_a_lane, Monoid::Max);
        TopValues::<f32, N, Const<2>>::value_line(&acc, slot)
    }

    #[cube(harness)]
    fn value_of_one_row<N: Size>(
        first: Vector<f32, N>,
        second: Vector<f32, N>,
        #[comptime] slot: usize,
    ) -> Vector<f32, N> {
        let index = Vector::new(0u32);
        let mut acc = TopValues::<f32, N, Const<2>>::identity(Monoid::Max);
        TopValues::<f32, N, Const<2>>::set_line(&mut acc, 0, first, index);
        TopValues::<f32, N, Const<2>>::set_line(&mut acc, 1, second, index);
        let row = TopValues::<f32, N, Const<2>>::one_row(acc, Monoid::Max);
        TopValues::<f32, N, Const<2>>::value_line(&row, slot)
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

    #[cube(harness)]
    fn index_kept<N: Size>(line: Vector<f32, N>) -> Vector<u32, N> {
        let a_row_a_lane = comptime!(1usize);
        let mut acc = TopValues::<f32, N, Const<2>>::identity(Monoid::Max);
        TopValues::<f32, N, Const<2>>::accumulate(&mut acc, line, 1, a_row_a_lane, Monoid::Max);
        TopValues::<f32, N, Const<2>>::index_line(&acc, 0)
    }

    #[test]
    fn no_index_is_kept() {
        let index = index_kept::harness::<TestRuntime, Const<2>>(&[1.0, 5.0]);
        assert_eq!(index, [0, 0]);
    }
}
