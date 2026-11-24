use super::base::UnitLinesWriteMicrokernel;
use crate::reduce::{ReduceAccumulator, ReduceOperation, ReduceWrites};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, cube, prelude::*};

/// The lines of `acc`, whose lanes are met, written by one unit to `out` and `indices`, each slot a
/// line `W` wide: one cell, or `acc`'s own line. `values` is what a mean divides by.
#[cube]
pub(crate) fn unit_lines_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    W: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    acc: C,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitLinesWriteMicrokernel,
) {
    let operation = microkernel.item().operation;
    let writes = microkernel.item().writes();
    let handed = indices.is_some();
    comptime!(assert!(
        writes == ReduceWrites::Values || operation.carrier().carries_indices(),
        "UnitLinesWriteMicrokernel: {operation:?} keeps no index to write"
    ));
    comptime!(assert!(
        handed == (writes == ReduceWrites::Both),
        "UnitLinesWriteMicrokernel: an output of indices is handed over exactly where they are \
         written beside the values"
    ));
    #[unroll]
    for slot in 0..operation.written_per_row() {
        let line =
            match writes {
                ReduceWrites::Values | ReduceWrites::Both => as_written::<Out, Acc, N, W>(
                    finished::<Acc, N>(C::value_line(&acc, slot), values, operation),
                ),
                ReduceWrites::Indices => as_written::<Out, u32, N, W>(C::index_line(&acc, slot)),
            };
        out.flat_mut::<Out, W>().write(slot, line);
        #[comptime]
        match indices.clone() {
            ComptimeOption::Some(mut tile) => tile.flat_mut::<Idx, W>().write(
                slot,
                as_written::<Idx, u32, N, W>(C::index_line(&acc, slot)),
            ),
            ComptimeOption::None => {}
        }
    }
}

/// `line` in `Out`, `W` wide: itself, or its first lane where `W` is one.
#[cube(harness)]
fn as_written<Out: Numeric, E: Numeric, N: Size, W: Size>(line: Vector<E, N>) -> Vector<Out, W> {
    let width = W::value();
    let lanes = N::value();
    match comptime!(width == lanes) {
        true => Vector::cast_from(line),
        false => Vector::cast_from(Vector::<E, W>::new(line.extract(0usize))),
    }
}

/// `line` as the row writes it: divided by its `values` for a mean, as it is otherwise.
#[cube]
fn finished<Acc: Numeric, N: Size>(
    line: Vector<Acc, N>,
    values: usize,
    #[comptime] operation: ReduceOperation,
) -> Vector<Acc, N> {
    match operation.divides_by_count() {
        true => divided::<Acc, N>(line, values),
        false => line,
    }
}

/// `line` divided by `values`, in its own element where that holds every count and every total
/// `line` can hold, in `f32` for a narrower one.
#[cube]
fn divided<Acc: Numeric, N: Size>(line: Vector<Acc, N>, values: usize) -> Vector<Acc, N> {
    let element = elem_type_of::<Acc>();
    match comptime!(divides_in_f32(element)) {
        true => Vector::cast_from(Vector::<f32, N>::cast_from(line) / Vector::new(values as f32)),
        false => line / Vector::new(Acc::cast_from(values)),
    }
}

/// Whether a total in `accumulator` is divided in `f32`: one of at most 16 bits, which a count
/// overflows, and `f32` itself.
fn divides_in_f32(accumulator: ElemType) -> bool {
    accumulator.size() < size_of::<f32>() || accumulator == ElemType::Float(FloatKind::F32)
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::TestRuntime;
    use cubecl::ir::IntKind;

    /// A line as wide as it is written keeps its lanes.
    #[test]
    fn a_line_is_written_whole() {
        let written =
            as_written::harness::<TestRuntime, i32, f32, Const<4>, Const<4>>(&[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(written, [1, 2, 3, 4]);
    }

    /// A line written as one cell writes its first lane.
    #[test]
    fn one_cell_is_the_lines_first_lane() {
        let written =
            as_written::harness::<TestRuntime, i32, f32, Const<4>, Const<1>>(&[5.0, 2.0, 3.0, 4.0]);
        assert_eq!(written, [5]);
    }

    /// A total in `i32` holds what no `f32` does, so it is divided as it is.
    #[test]
    fn an_i32_total_is_divided_in_its_own_element() {
        assert!(!divides_in_f32(ElemType::Int(IntKind::I32)));
    }

    /// So is a total in `f64`.
    #[test]
    fn an_f64_total_is_divided_in_its_own_element() {
        assert!(!divides_in_f32(ElemType::Float(FloatKind::F64)));
    }

    /// A count overflows a half, so a total in one is divided in `f32`.
    #[test]
    fn a_half_total_is_divided_in_f32() {
        assert!(divides_in_f32(ElemType::Float(FloatKind::F16)));
    }
}
