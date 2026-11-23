//! The values themselves, pointwise on the line: a sum, a product, a max, a mean.

use crate::reduce::ReduceAccumulator;
use cubecl::tile::Monoid;
use cubecl::{self as cubecl, cube, prelude::*};

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
