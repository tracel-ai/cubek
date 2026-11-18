//! What every carrier does with one line: read it, index its lanes, write it out.

use crate::reduce::{ReadMap, ReduceLayout, ReduceWrites};
use cubecl::tile::{Indexed, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// One lane's value and index, what an [`Indexed`] row's lanes reduce to.
pub(crate) type OneIndexed<Acc> = Indexed<Acc, Const<1>>;

/// `line` as `map` reads it.
#[cube]
pub(crate) fn read_mapped<Acc: Numeric, N: Size>(
    line: Vector<Acc, N>,
    #[comptime] map: ReadMap,
) -> Vector<Acc, N> {
    match map {
        ReadMap::AsRead => line,
        ReadMap::Abs => line.abs(),
        ReadMap::NonZero => select_many(
            line.not_equal(&Vector::new(Acc::from_int(0))),
            Vector::new(Acc::from_int(1)),
            Vector::new(Acc::from_int(0)),
        ),
    }
}

/// The index of each lane of the line read `line_index` lines into the row: along the row,
/// consecutive values; across rows, the line's own position, each lane in its own row.
#[cube]
pub(crate) fn lane_indices<N: Size>(
    line_index: usize,
    #[comptime] layout: ReduceLayout,
) -> Vector<u32, N> {
    match layout {
        ReduceLayout::Along => consecutive_indices::<N>(line_index),
        ReduceLayout::Across => Vector::new(line_index as u32),
    }
}

/// The indices of the `N` consecutive values of the line read `line_index` lines into a row.
#[cube]
fn consecutive_indices<N: Size>(line_index: usize) -> Vector<u32, N> {
    let width = N::value();
    let mut indices = Vector::<u32, N>::new((line_index * width) as u32);
    #[unroll]
    for lane in 0..width {
        indices.insert(lane, (line_index * width + lane) as u32);
    }
    indices
}

/// Line `at` of what an owner writes, `value` and its `index`, written as `writes` says: one of
/// them to `out`, or the value to `out` and the index to `indices`, which nothing else writes.
#[cube]
pub(crate) fn write_line<Out: Numeric, Idx: Numeric, Acc: Numeric, W: Size>(
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    at: usize,
    value: Vector<Acc, W>,
    index: Vector<u32, W>,
    #[comptime] writes: ReduceWrites,
) {
    let mut lines = out.flat_mut::<Out, W>();
    match writes {
        ReduceWrites::Values => lines.write(at, Vector::cast_from(value)),
        ReduceWrites::Indices => lines.write(at, Vector::cast_from(index)),
        ReduceWrites::Both => write_both::<Out, Idx, Acc, W>(out, indices, at, value, index),
    }
}

/// Line `at`, `value` to `out` and its `index` to `indices`.
#[cube]
fn write_both<Out: Numeric, Idx: Numeric, Acc: Numeric, W: Size>(
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    at: usize,
    value: Vector<Acc, W>,
    index: Vector<u32, W>,
) {
    let mut lines = out.flat_mut::<Out, W>();
    lines.write(at, Vector::cast_from(value));
    let mut index_tile = indices.clone().unwrap();
    let mut index_lines = index_tile.flat_mut::<Idx, W>();
    index_lines.write(at, Vector::cast_from(index));
}

/// The values a row of `lines` lines holds: `N` a line along the row, one across rows.
#[cube]
pub(crate) fn row_values<N: Size>(lines: usize, #[comptime] layout: ReduceLayout) -> usize {
    match layout {
        ReduceLayout::Along => lines * N::value(),
        ReduceLayout::Across => lines,
    }
}
