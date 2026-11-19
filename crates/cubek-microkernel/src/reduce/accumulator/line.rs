//! What every carrier does with one line: read it, index its lanes, write it out.

use crate::reduce::{REDUCED, ReadMap, ReduceAccumulator, ReduceWorkItem, ReduceWrites};
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

/// The lanes of a line of `input` that hold one row's values: all of them where its lines run along
/// the row, one where each lane is its own row's. What one load of the tile reaches along
/// [`REDUCED`], as the binding laid it.
#[cube]
pub fn row_lanes(input: &Tile) -> comptime_type!(usize) {
    let load = input.vector_tile();
    comptime!(load.extent_along(REDUCED))
}

/// `read`, a line of a row in `In`, `row_lanes` of its lanes holding one row's values, mapped as
/// `item`'s operation reads it and accumulated in `Acc` into `acc` as line `index` of its row.
#[cube]
pub(crate) fn accumulate_read<In: Numeric, Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    acc: &mut C,
    read: Vector<In, N>,
    index: usize,
    #[comptime] row_lanes: usize,
    #[comptime] item: ReduceWorkItem,
) {
    let operation = item.operation;
    let mapped = read_mapped::<Acc, N>(Vector::<Acc, N>::cast_from(read), operation.read_map());
    C::accumulate(acc, mapped, index, row_lanes, operation.monoid());
}

/// The lines of `input` from `first` on, every `stride`-th up to `count`, accumulated from the
/// identity, line `l` as line `offset + l` of its row: one unit's share of a row its plane or its
/// cube reads together.
#[cube]
pub(crate) fn accumulate_strided<
    In: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    input: &Tile,
    first: usize,
    #[comptime] stride: usize,
    count: usize,
    offset: usize,
    #[comptime] item: ReduceWorkItem,
) -> C {
    let lines = input.flat::<In, N>();
    let row_lanes = row_lanes(input);
    let mut acc = C::identity(item.operation.monoid());
    for step in 0..count.div_ceil(stride) {
        let line = first + step * stride;
        if line < count {
            accumulate_read::<In, Acc, N, C>(
                &mut acc,
                lines.read(line),
                offset + line,
                row_lanes,
                item,
            );
        }
    }
    acc
}

/// The index of each lane of the line read `line_index` lines into the row, `row_lanes` of its
/// lanes holding one row's values: along the row, consecutive values; across rows, the line's own
/// position, each lane in its own row.
#[cube]
pub(crate) fn lane_indices<N: Size>(
    line_index: usize,
    #[comptime] row_lanes: usize,
) -> Vector<u32, N> {
    match row_lanes > 1 {
        true => consecutive_indices_of_line::<N>(line_index),
        false => Vector::new(line_index as u32),
    }
}

/// The indices of the `N` consecutive values of the line read `line_index` lines into a row.
#[cube]
fn consecutive_indices_of_line<N: Size>(line_index: usize) -> Vector<u32, N> {
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

/// The values a row of `lines` lines holds, `row_lanes` of each line's lanes holding it.
#[cube]
pub(crate) fn row_values(lines: usize, #[comptime] row_lanes: usize) -> usize {
    lines * row_lanes
}
