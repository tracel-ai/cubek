use crate::reduce::write::{UnitLinesWriteMicrokernel, unit_lines_write_microkernel};
use crate::reduce::{Accumulable, ReduceWorkItem};
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, IntoRuntime, cube, prelude::*};

/// One unit writing what its carrier keeps to its window of the output: each slot's line, the
/// values divided by the row's length for a mean, the indices beside them where its item writes
/// both. Where a line's lanes hold one row, they are combined first and the row
/// written as one cell a slot; where each lane is its own row's, the lines are written as they are.
///
/// Every reduce microkernel hands it the carrier its scope met and the window of its owner's rows.
#[derive(CubeType, IntoRuntime, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitCarrierWriteMicrokernel {
    #[cube(comptime)]
    item: ReduceWorkItem,
    /// The lanes of a line that hold one row's values: the line's width along the row, one
    /// across rows.
    #[cube(comptime)]
    row_lanes: usize,
}

impl UnitCarrierWriteMicrokernel {
    pub fn new(item: ReduceWorkItem, row_lanes: usize) -> Self {
        Self { item, row_lanes }
    }
}

#[cube]
impl UnitCarrierWriteMicrokernel {
    /// `acc` written to `out` and `indices`, `values` the values each of its rows holds
    /// (`unit_carrier_write_microkernel`).
    ///
    /// # Assumptions
    ///
    /// - `indices` holds a tile exactly where the item writes the indices beside the values
    ///   ([`ReduceWrites::Both`](crate::reduce::ReduceWrites::Both)).
    ///
    /// # Panics
    ///
    /// When the kernel compiles, where an Assumption, or one of [`ReduceWorkItem`](crate::reduce::ReduceWorkItem)'s, is broken.
    pub fn write<Out: Numeric, Idx: Numeric, Acc: Numeric, N: Size, C: Accumulable<Acc, N>>(
        &self,
        acc: C,
        out: &mut Tile,
        indices: &ComptimeOption<Tile>,
        values: usize,
    ) {
        unit_carrier_write_microkernel::<Out, Idx, Acc, N, C>(
            acc,
            out,
            indices,
            values,
            comptime!(UnitCarrierWriteMicrokernel {
                item: self.item,
                row_lanes: self.row_lanes
            }),
        )
    }
}

/// `acc` written by one unit to `out`, its window of the output, and `indices` beside it, slot by
/// slot, `values` the values each of its rows holds. Where a line's lanes hold one row
/// (`row_lanes` above one), the lanes are combined into
/// the row's carrier first and each slot written as one cell; the whole line, a row a lane,
/// otherwise.
#[cube]
pub(crate) fn unit_carrier_write_microkernel<
    Out: Numeric,
    Idx: Numeric,
    Acc: Numeric,
    N: Size,
    C: Accumulable<Acc, N>,
>(
    acc: C,
    out: &mut Tile,
    indices: &ComptimeOption<Tile>,
    values: usize,
    #[comptime] microkernel: UnitCarrierWriteMicrokernel,
) {
    let item = microkernel.item;
    match microkernel.row_lanes > 1 {
        true => unit_lines_write_microkernel::<Out, Idx, Acc, N, Const<1>, C>(
            C::one_row(acc, item.operation.monoid()),
            out,
            indices,
            values,
            comptime!(UnitLinesWriteMicrokernel::new(item)),
        ),
        false => unit_lines_write_microkernel::<Out, Idx, Acc, N, N, C>(
            acc,
            out,
            indices,
            values,
            comptime!(UnitLinesWriteMicrokernel::new(item)),
        ),
    }
}

#[cfg(test)]
mod tests {

    use crate::reduce::testing::UnitCarrierMicrokernelLauncher;

    /// Rows of four values, one line each: its lanes combined into the row's cell.
    mod along_one_line_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitCarrierMicrokernelLauncher, Along, 4, 4, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
        ]);
    }

    /// Rows of one value, four rows a line: the line written as it is, a row a lane.
    mod across_one_value_four_wide {
        use super::*;
        use crate::reduce::ReduceOperation;
        reduce_tests!(UnitCarrierMicrokernelLauncher, Across, 1, 4, Plain, [
            sum: ReduceOperation::Sum,
            prod: ReduceOperation::Prod,
            mean: ReduceOperation::Mean,
            max: ReduceOperation::Max,
            min: ReduceOperation::Min,
        ]);
    }
}
