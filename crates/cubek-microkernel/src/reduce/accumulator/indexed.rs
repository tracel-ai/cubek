//! A value with the index it was read at: the argmax and the argmin.

use crate::reduce::{OneIndexed, ReduceAccumulator, lane_indices, stage_slot};
use cubecl::tile::{Carrier, Indexed, Monoid};
use cubecl::{self as cubecl, cube, prelude::*};

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

    fn across_planes(
        acc: Self,
        column: usize,
        stages: bool,
        #[comptime] planes: usize,
        #[comptime] columns: usize,
        #[comptime] monoid: Monoid,
    ) -> Self {
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(planes * columns);
        let mut indices = Shared::<[Vector<u32, N>]>::new_slice(planes * columns);
        if stages {
            stage_plane_partial::<Acc, N>(
                &mut values,
                &mut indices,
                stage_slot(PLANE_POS as usize, column, columns),
                acc,
            );
        }

        sync_cube();

        let mut total = Indexed::<Acc, N>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            let slot = Indexed::<Acc, N> {
                value: values[stage_slot(plane, column, columns)],
                index: indices[stage_slot(plane, column, columns)],
            };
            total = Indexed::<Acc, N>::combine(total, slot, monoid);
        }

        sync_cube();
        total
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

/// A plane's partial, `acc`, staged in slot `slot` of `values` and `indices`.
#[cube]
fn stage_plane_partial<Acc: Numeric, N: Size>(
    values: &mut Shared<[Vector<Acc, N>]>,
    indices: &mut Shared<[Vector<u32, N>]>,
    slot: usize,
    acc: Indexed<Acc, N>,
) {
    values[slot] = acc.value;
    indices[slot] = acc.index;
}
