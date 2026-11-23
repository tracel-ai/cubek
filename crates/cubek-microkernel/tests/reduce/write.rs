//! The write every reduce microkernel ends on, alone: a unit's carrier of values written to its
//! window of the output. The carrier is the unit's line of the input as it is, so the operations
//! are those whose carrier is the values themselves, read as they are.

use crate::launch::reduce::TestedMicrokernel;

/// The microkernel under test.
const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::UnitCarrierWrite;

/// Rows of four values, one line each: its lanes combined into the row's cell.
mod along_one_line_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(MICROKERNEL, Along, 4, 4, Plain, [
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
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(MICROKERNEL, Across, 1, 4, Plain, [
        sum: ReduceOperation::Sum,
        prod: ReduceOperation::Prod,
        mean: ReduceOperation::Mean,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
    ]);
}
