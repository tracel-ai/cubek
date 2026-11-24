//! The values themselves, pointwise on the line: a sum, a product, a max, a mean.

use crate::reduce::{ReduceAccumulator, ReduceCarrierFamily};
use cubecl::tile::Monoid;
use cubecl::{self as cubecl, cube, prelude::*};

/// [`ReduceCarrier::Values`](crate::reduce::ReduceCarrier::Values): its carrier is [`Vector`].
pub struct ValuesReduceCarrierFamily;

impl ReduceCarrierFamily for ValuesReduceCarrierFamily {
    type Carrier<Acc: Numeric, N: Size, K: Size> = Vector<Acc, N>;
}

#[cube]
impl<Acc: Numeric, N: Size> ReduceAccumulator<Acc, N> for Vector<Acc, N> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        _line_index: usize,
        #[comptime] _row_lanes: usize,
        #[comptime] monoid: Monoid,
    ) {
        *acc = monoid.combine::<Vector<Acc, N>>(*acc, line);
    }

    fn value_line(acc: &Self, _slot: usize) -> Vector<Acc, N> {
        *acc
    }

    fn index_line(_acc: &Self, _slot: usize) -> Vector<u32, N> {
        Vector::new(0u32)
    }

    fn set_line(acc: &mut Self, _slot: usize, value: Vector<Acc, N>, _index: Vector<u32, N>) {
        *acc = value;
    }

    fn one_row(acc: Self, #[comptime] monoid: Monoid) -> Self {
        Vector::new(Monoid::reduce::<Acc, N>(acc, N::value(), monoid))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::TestRuntime;

    /// `acc` with `line` accumulated under `Sum`.
    #[cube(harness)]
    fn accumulated<N: Size>(acc: Vector<f32, N>, line: Vector<f32, N>) -> Vector<f32, N> {
        let mut acc = acc;
        let row_lanes = N::value();
        Vector::<f32, N>::accumulate(&mut acc, line, 0, row_lanes, Monoid::Sum);
        acc
    }

    #[cube(harness)]
    fn value_line<N: Size>(acc: Vector<f32, N>) -> Vector<f32, N> {
        Vector::<f32, N>::value_line(&acc, 0)
    }

    #[cube(harness)]
    fn index_line<N: Size>(acc: Vector<f32, N>) -> Vector<u32, N> {
        Vector::<f32, N>::index_line(&acc, 0)
    }

    /// `acc` with its line set to `value`.
    #[cube(harness)]
    fn set<N: Size>(acc: Vector<f32, N>, value: Vector<f32, N>) -> Vector<f32, N> {
        let mut acc = acc;
        Vector::<f32, N>::set_line(&mut acc, 0, value, Vector::new(0u32));
        acc
    }

    #[cube(harness)]
    fn one_row<N: Size>(acc: Vector<f32, N>) -> Vector<f32, N> {
        Vector::<f32, N>::one_row(acc, Monoid::Sum)
    }

    #[test]
    fn a_line_joins_lane_by_lane() {
        let acc = accumulated::harness::<TestRuntime, Const<4>>(
            &[1.0, 2.0, 3.0, 4.0],
            &[10.0, 20.0, 30.0, 40.0],
        );
        assert_eq!(acc, [11.0, 22.0, 33.0, 44.0]);
    }

    #[test]
    fn its_line_is_itself() {
        let line = value_line::harness::<TestRuntime, Const<4>>(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(line, [1.0, 2.0, 3.0, 4.0]);
    }

    #[test]
    fn it_keeps_no_index() {
        let line = index_line::harness::<TestRuntime, Const<4>>(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(line, [0, 0, 0, 0]);
    }

    #[test]
    fn its_line_set_replaces_it() {
        let acc =
            set::harness::<TestRuntime, Const<4>>(&[1.0, 2.0, 3.0, 4.0], &[5.0, 6.0, 7.0, 8.0]);
        assert_eq!(acc, [5.0, 6.0, 7.0, 8.0]);
    }

    #[test]
    fn one_row_is_every_lanes_total_in_every_lane() {
        let acc = one_row::harness::<TestRuntime, Const<4>>(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(acc, [10.0, 10.0, 10.0, 10.0]);
    }
}
