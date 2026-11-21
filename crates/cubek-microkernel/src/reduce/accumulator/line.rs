//! What every carrier does with one line: read it, index its lanes, meet across a column.

use crate::reduce::{ReadMap, ReduceAccumulator, ReduceWorkItem};
use cubecl::tile::space::UnitShare;
use cubecl::tile::{Indexed, Monoid};
use cubecl::{self as cubecl, cube, prelude::*};

/// One lane's value and index, what an [`Indexed`] row's lanes reduce to.
pub(crate) type OneIndexed<Acc> = Indexed<Acc, Const<1>>;

/// `line` as `map` reads it.
#[cube(harness)]
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

/// `acc` met across the `per_column` units of its column within the plane, which hold the low
/// bits of the lane: every unit of the column is left with the plane's share of its total.
#[cube]
pub(crate) fn across_column<Acc: Numeric, N: Size, C: ReduceAccumulator<Acc, N>>(
    acc: C,
    #[comptime] per_column: usize,
    #[comptime] monoid: Monoid,
) -> C {
    let column = comptime!(UnitShare::Group {
        unit_bits: per_column - 1
    });
    C::across(acc, column, monoid)
}

/// The index of each lane of the line read `line_index` lines into the row, `row_lanes` of its
/// lanes holding one row's values: along the row, consecutive values; across rows, the line's own
/// position, each lane in its own row.
#[cube(harness)]
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

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::TestRuntime;

    /// A negative, a zero and a fraction: read as they are.
    #[test]
    fn as_read_keeps_every_value() {
        let read = read_mapped::harness::<TestRuntime, f32, Const<4>>(
            &[-2.0, 0.0, 0.5, 3.0],
            ReadMap::AsRead,
        );
        assert_eq!(read, [-2.0, 0.0, 0.5, 3.0]);
    }

    /// A negative, a zero and a fraction: each read as its magnitude.
    #[test]
    fn abs_reads_each_magnitude() {
        let read = read_mapped::harness::<TestRuntime, f32, Const<4>>(
            &[-2.0, 0.0, 0.5, -0.0],
            ReadMap::Abs,
        );
        assert_eq!(read, [2.0, 0.0, 0.5, 0.0]);
    }

    /// A zero, a negative, a fraction and a negative zero: one wherever the value is not zero.
    #[test]
    fn non_zero_reads_one_where_the_value_is_not_zero() {
        let read = read_mapped::harness::<TestRuntime, f32, Const<4>>(
            &[0.0, -2.0, 0.5, -0.0],
            ReadMap::NonZero,
        );
        assert_eq!(read, [0.0, 1.0, 1.0, 0.0]);
    }

    /// The third line of a row whose four lanes all hold its values: the row's values 8 to 11.
    #[test]
    fn a_line_along_a_row_indexes_its_values() {
        let indices = lane_indices::harness::<TestRuntime, Const<4>>(2, 4);
        assert_eq!(indices, [8, 9, 10, 11]);
    }

    /// The third line where each lane is its own row's: every lane at its row's third value.
    #[test]
    fn a_line_across_rows_indexes_its_position() {
        let indices = lane_indices::harness::<TestRuntime, Const<4>>(2, 1);
        assert_eq!(indices, [2, 2, 2, 2]);
    }
}
