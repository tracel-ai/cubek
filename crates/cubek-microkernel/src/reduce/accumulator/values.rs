//! The values themselves, pointwise on the line: a sum, a product, a max, a mean.

use crate::reduce::{ReduceAccumulator, stage_slot};
use cubecl::tile::{Carrier, Monoid};
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

    fn across_planes(
        acc: Self,
        column: usize,
        stages: bool,
        #[comptime] planes: usize,
        #[comptime] columns: usize,
        #[comptime] monoid: Monoid,
    ) -> Self {
        let mut slots = Shared::<[Vector<Acc, N>]>::new_slice(planes * columns);
        if stages {
            slots[stage_slot(PLANE_POS as usize, column, columns)] = acc;
        }

        sync_cube();

        let mut total = Vector::<Acc, N>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            total =
                monoid.combine::<Vector<Acc, N>>(total, slots[stage_slot(plane, column, columns)]);
        }

        sync_cube();
        total
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
