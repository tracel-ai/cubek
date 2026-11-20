//! The values themselves, pointwise on the line: a sum, a product, a max, a mean.

use crate::reduce::{ReduceAccumulator, ReduceOperation, ReduceWrites};
use cubecl::tile::{Carrier, Monoid, Tile};
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
            slots[PLANE_POS as usize * columns + column] = acc;
        }

        sync_cube();

        let mut total = Vector::<Acc, N>::identity(monoid);
        #[unroll]
        for plane in 0..planes {
            total = monoid.combine::<Vector<Acc, N>>(total, slots[plane * columns + column]);
        }

        sync_cube();
        total
    }

    fn store(acc: Self, values: &mut Tile, _indices: &ComptimeOption<Tile>, part: usize) {
        let mut lines = values.flat_mut::<Acc, N>();
        lines.write(part, acc);
    }

    fn load(values: &Tile, _indices: &ComptimeOption<Tile>, part: usize) -> Self {
        values.flat::<Acc, N>().read(part)
    }

    fn write<Out: Numeric, Idx: Numeric>(
        acc: Self,
        out: &mut Tile,
        _indices: &ComptimeOption<Tile>,
        count: usize,
        #[comptime] row_lanes: usize,
        #[comptime] operation: ReduceOperation,
        #[comptime] writes: ReduceWrites,
    ) {
        comptime!(assert!(
            writes == ReduceWrites::Values,
            "ReduceAccumulator: {operation:?} accumulates the values themselves, which carry no index to write"
        ));

        match row_lanes > 1 {
            true => write_row::<Acc, Out, N>(acc, out, count, operation),
            false => write_rows::<Acc, Out, N>(acc, out, count, operation),
        }
    }
}

/// `acc`, whose lanes hold one row, reduced to the row's cell of `out`, divided by `count` for a
/// mean.
#[cube]
fn write_row<Acc: Numeric, Out: Numeric, N: Size>(
    acc: Vector<Acc, N>,
    out: &mut Tile,
    count: usize,
    #[comptime] operation: ReduceOperation,
) {
    let mut total = Monoid::reduce::<Acc, N>(acc, N::value(), operation.monoid());
    if operation.divides_by_count() {
        total = Acc::cast_from(f32::cast_from(total) / count as f32);
    }

    let mut cells = out.flat_mut::<Out, Const<1>>();
    cells.write(0, Vector::cast_from(total));
}

/// `acc`, each lane a row's total, written as `out`'s line of rows, divided by `count` for a mean.
#[cube]
fn write_rows<Acc: Numeric, Out: Numeric, N: Size>(
    acc: Vector<Acc, N>,
    out: &mut Tile,
    count: usize,
    #[comptime] operation: ReduceOperation,
) {
    let mut total = acc;
    if operation.divides_by_count() {
        total = Vector::cast_from(Vector::<f32, N>::cast_from(total) / Vector::new(count as f32));
    }

    let mut lines = out.flat_mut::<Out, N>();
    lines.write(0, Vector::cast_from(total));
}
