use super::base::{UnitCarrierWriteMicrokernel, UnitWalkWriteMicrokernel};
use crate::reduce::{REDUCED, ReduceAccumulator, ReduceOperation, ReduceWrites};
use cubecl::ir::{ElemType, FloatKind};
use cubecl::tile::{Region, Tile};
use cubecl::{self as cubecl, cube, prelude::*};

/// `acc` written by one unit to `out`, its window of the output, and `indices` beside it, slot by
/// slot, `values` the values each of its rows held. Where a line's lanes hold one row
/// ([`row_lanes`](UnitCarrierWriteMicrokernel::row_lanes) above one), the lanes are combined into
/// the row's carrier first and each slot written as one cell.
#[cube]
pub(crate) fn unit_carrier_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N>,
>(
    acc: C,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitCarrierWriteMicrokernel,
) {
    let operation = microkernel.item().operation;
    let writes = microkernel.item().writes();
    let one_row = microkernel.row_lanes() > 1;
    comptime!(assert!(
        writes == ReduceWrites::Values || operation.carrier().carries_indices(),
        "UnitCarrierWriteMicrokernel: {operation:?} keeps no index to write"
    ));
    let acc = match one_row {
        true => C::one_row(acc, operation.monoid()),
        false => acc,
    };

    #[unroll]
    for slot in 0..operation.written_per_row() {
        match writes {
            ReduceWrites::Values => {
                let value = finished::<Acc, N>(C::value_line(&acc, slot), values, operation);
                write_slot::<Out, Acc, N>(out, slot, value, one_row);
            }
            ReduceWrites::Indices => {
                write_slot::<Out, u32, N>(out, slot, C::index_line(&acc, slot), one_row);
            }
            ReduceWrites::Both => {
                let value = finished::<Acc, N>(C::value_line(&acc, slot), values, operation);
                write_slot::<Out, Acc, N>(out, slot, value, one_row);
                let mut index_tile = indices.clone().unwrap();
                write_slot::<Idx, u32, N>(
                    &mut index_tile,
                    slot,
                    C::index_line(&acc, slot),
                    one_row,
                );
            }
        }
    }
}

/// `total`, a column of rows' carrier, written to `out` and `indices` by the column's unit at the
/// start of the reduced dimension in `region`'s walk, at that unit's cell, one write a column, a
/// row a lane, `values` the values each row held.
#[cube]
pub(crate) fn unit_walk_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: ReduceAccumulator<Acc, N> + Clone,
>(
    region: &Region,
    total: C,
    out: &Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitWalkWriteMicrokernel,
) {
    let a_row_a_lane = comptime!(1usize);
    for unit in region.walk().region(0).leaves() {
        if unit.origin(REDUCED) == 0 {
            let mut cell_out = out.at(&unit);
            let cell_indices = indices.clone().map(|tile| tile.at(&unit));
            unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
                total.clone(),
                &mut cell_out,
                &cell_indices,
                values,
                comptime!(UnitCarrierWriteMicrokernel::new(
                    microkernel.item(),
                    a_row_a_lane
                )),
            );
        }
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

/// `line` written as `tile`'s line `slot` in `Out`: its first lane alone, as one cell, where it
/// holds `one_row`, whose lanes all hold the row's cell; the whole line, a row a lane, otherwise.
#[cube]
fn write_slot<Out: Numeric, E: Numeric, N: Size>(
    tile: &mut Tile,
    slot: usize,
    line: Vector<E, N>,
    #[comptime] one_row: bool,
) {
    match one_row {
        true => {
            let mut cells = tile.flat_mut::<Out, Const<1>>();
            cells.write(
                slot,
                Vector::cast_from(Vector::<E, Const<1>>::new(line.extract(0usize))),
            );
        }
        false => {
            let mut lines = tile.flat_mut::<Out, N>();
            lines.write(slot, Vector::cast_from(line));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl::ir::IntKind;

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
