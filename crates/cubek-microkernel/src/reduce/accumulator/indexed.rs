//! A value with the index it was read at: the argmax and the argmin.

use crate::reduce::{OneIndexed, ReduceAccumulator, ReduceCarrierFamily, lane_indices};
use cubecl::tile::{Carrier, Indexed, Monoid};
use cubecl::{self as cubecl, cube, prelude::*};

/// [`ReduceCarrier::Indexed`](crate::reduce::ReduceCarrier::Indexed): its carrier is [`Indexed`].
pub struct IndexedReduceCarrierFamily;

impl ReduceCarrierFamily for IndexedReduceCarrierFamily {
    type Carrier<Acc: Numeric, N: Size, K: Size> = Indexed<Acc, N>;
}

#[cube]
impl<Acc: Numeric, N: Size> ReduceAccumulator<Acc, N> for Indexed<Acc, N> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] row_lanes: usize,
        #[comptime] monoid: Monoid,
    ) {
        let read = Indexed::<Acc, N> {
            value: line,
            index: lane_indices::<N>(line_index, row_lanes),
        };
        *acc = Indexed::<Acc, N>::combine_later(*acc, read, monoid);
    }

    fn value_line(acc: &Self, _slot: usize) -> Vector<Acc, N> {
        acc.value
    }

    fn index_line(acc: &Self, _slot: usize) -> Vector<u32, N> {
        acc.index
    }

    fn set_line(acc: &mut Self, _slot: usize, value: Vector<Acc, N>, index: Vector<u32, N>) {
        acc.value = value;
        acc.index = index;
    }

    /// Its lanes reduced one by one as one-lane carriers under `monoid`.
    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self {
        let mut best = OneIndexed::<Acc>::identity(monoid);
        #[unroll]
        for lane in 0..N::value() {
            let one = OneIndexed::<Acc> {
                value: Vector::new(acc.value.extract(lane)),
                index: Vector::new(acc.index.extract(lane)),
            };
            best = OneIndexed::<Acc>::combine(best, one, monoid);
        }
        Indexed::<Acc, N> {
            value: Vector::new(best.value.extract(0usize)),
            index: Vector::new(best.index.extract(0usize)),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::TestRuntime;

    /// The index of the best of `first` and, a line later, `second`, each lane its own row.
    #[cube(harness)]
    fn index_of_best<N: Size>(first: Vector<f32, N>, second: Vector<f32, N>) -> Vector<u32, N> {
        let a_row_a_lane = comptime!(1usize);
        let mut acc = Indexed::<f32, N>::identity(Monoid::Max);
        Indexed::<f32, N>::accumulate(&mut acc, first, 0, a_row_a_lane, Monoid::Max);
        Indexed::<f32, N>::accumulate(&mut acc, second, 1, a_row_a_lane, Monoid::Max);
        Indexed::<f32, N>::index_line(&acc, 0)
    }

    /// The value line of a carrier whose line was set to `value`.
    #[cube(harness)]
    fn value_set<N: Size>(value: Vector<f32, N>, index: Vector<u32, N>) -> Vector<f32, N> {
        let mut acc = Indexed::<f32, N>::identity(Monoid::Max);
        Indexed::<f32, N>::set_line(&mut acc, 0, value, index);
        Indexed::<f32, N>::value_line(&acc, 0)
    }

    /// The index the lanes of `value` and `index` reduce to as one row, in every lane.
    #[cube(harness)]
    fn index_of_one_row<N: Size>(value: Vector<f32, N>, index: Vector<u32, N>) -> Vector<u32, N> {
        let mut acc = Indexed::<f32, N>::identity(Monoid::Max);
        Indexed::<f32, N>::set_line(&mut acc, 0, value, index);
        let row = Indexed::<f32, N>::one_row(acc, Monoid::Max);
        Indexed::<f32, N>::index_line(&row, 0)
    }

    /// A tie keeps the earlier line's index.
    #[test]
    fn the_best_lines_index_is_kept() {
        let index = index_of_best::harness::<TestRuntime, Const<4>>(
            &[1.0, 5.0, 3.0, 2.0],
            &[4.0, 2.0, 3.0, 9.0],
        );
        assert_eq!(index, [1, 0, 0, 1]);
    }

    #[test]
    fn the_value_set_is_its_line() {
        let value =
            value_set::harness::<TestRuntime, Const<4>>(&[1.0, 2.0, 3.0, 4.0], &[7, 6, 5, 4]);
        assert_eq!(value, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn one_row_is_the_best_lanes_index_in_every_lane() {
        let index = index_of_one_row::harness::<TestRuntime, Const<4>>(
            &[1.0, 9.0, 3.0, 4.0],
            &[7, 6, 5, 4],
        );
        assert_eq!(index, [6, 6, 6, 6]);
    }
}
