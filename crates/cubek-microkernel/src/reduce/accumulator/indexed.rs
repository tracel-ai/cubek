//! A value with the index it was read at: the argmax and the argmin.

use crate::reduce::{
    OneIndexed, ReduceAccumulator, ReduceLayout, ReduceOperation, ReduceWrites, lane_indices,
    write_line,
};
use cubecl::tile::{Carrier, Indexed, Monoid, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

#[cube]
impl<Acc: Numeric, N: Size> ReduceAccumulator<Acc, N> for Indexed<Acc, N> {
    fn accumulate(
        acc: &mut Self,
        line: Vector<Acc, N>,
        line_index: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] monoid: Monoid,
    ) {
        let read = Indexed::<Acc, N> {
            value: line,
            index: lane_indices::<N>(line_index, layout),
        };
        *acc = Indexed::<Acc, N>::combine(*acc, read, monoid);
    }

    fn across_planes(acc: Self, #[comptime] planes: usize, #[comptime] monoid: Monoid) -> Self {
        let mut values = Shared::<[Vector<Acc, N>]>::new_slice(planes);
        let mut indices = Shared::<[Vector<u32, N>]>::new_slice(planes);
        if UNIT_POS_PLANE == 0 {
            stage_plane_partial::<Acc, N>(&mut values, &mut indices, acc);
        }

        sync_cube();

        let mut total = Indexed::<Acc, N>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            let slot = Indexed::<Acc, N> {
                value: values[plane],
                index: indices[plane],
            };
            total = Indexed::<Acc, N>::combine(total, slot, monoid);
        }

        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, indices: &ComptimeOption<Tile>, part: usize) {
        let mut value_lines = values.flat_mut::<Acc, N>();
        value_lines.write(part, acc.value);
        let mut index_tile = indices.clone().unwrap();
        let mut index_lines = index_tile.flat_mut::<u32, N>();
        index_lines.write(part, acc.index);
    }

    fn load(values: &Tile, indices: &ComptimeOption<Tile>, part: usize) -> Self {
        let index_tile = indices.clone().unwrap();
        Indexed::<Acc, N> {
            value: values.flat::<Acc, N>().read(part),
            index: index_tile.flat::<u32, N>().read(part),
        }
    }

    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        _count: usize,
        #[comptime] layout: ReduceLayout,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        match layout {
            ReduceLayout::Along => {
                write_row::<Out, Idx, Acc, N>(acc, out, indices, operation.monoid(), writes)
            }
            ReduceLayout::Across => {
                write_line::<Out, Idx, Acc, N>(out, indices, 0, acc.value, acc.index, writes)
            }
        }
    }
}

/// This plane's partial, `acc`, staged in its slot of `values` and `indices`.
#[cube]
fn stage_plane_partial<Acc: Numeric, N: Size>(
    values: &mut Shared<[Vector<Acc, N>]>,
    indices: &mut Shared<[Vector<u32, N>]>,
    acc: Indexed<Acc, N>,
) {
    values[PLANE_POS as usize] = acc.value;
    indices[PLANE_POS as usize] = acc.index;
}

/// `acc`, whose lanes hold one row, reduced lane by lane as one-lane carriers under `monoid` and
/// written as `writes` says.
#[cube]
fn write_row<Out: Numeric, Idx: Numeric, Acc: Numeric, N: Size>(
    acc: Indexed<Acc, N>,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    #[comptime] monoid: Monoid,
    #[comptime] writes: ReduceWrites,
) {
    let mut best = OneIndexed::<Acc>::identity(monoid);
    #[unroll]
    for lane in 0..N::value() {
        let one = OneIndexed::<Acc> {
            value: Vector::new(acc.value.extract(lane)),
            index: Vector::new(acc.index.extract(lane)),
        };
        best = OneIndexed::<Acc>::combine(best, one, monoid);
    }

    write_line::<Out, Idx, Acc, Const<1>>(out, indices, 0, best.value, best.index, writes);
}
