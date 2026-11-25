//! The writes other microkernels share, alone: a unit's carrier of values written. The carrier is
//! the unit's line of the input as it is, so the operations are those whose carrier is the values
//! themselves, read as they are.

use crate::launch::reduce::TestedMicrokernel;

/// The write every reduce microkernel ends on, to the unit's window of the output.
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

/// The write after a walked read, a plane's units across two lines of rows.
const WALK: TestedMicrokernel = TestedMicrokernel::UnitWalkWrite { columns: 2 };

/// The store of a partial.
const PARTIAL: TestedMicrokernel = TestedMicrokernel::UnitPartialStore;

/// Each column's first unit's line of rows 300 values long, written as it is, a row a lane.
mod walk_across_300_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(WALK, Across, 300, 4, Plain, [
        sum: ReduceOperation::Sum,
        mean: ReduceOperation::Mean,
        max: ReduceOperation::Max,
    ]);
}

/// A partial of four rows a line, stored as it is.
mod partial_across_one_value_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(PARTIAL, Across, 1, 4, Plain, [
        sum: ReduceOperation::Sum,
        max: ReduceOperation::Max,
    ]);
}

/// Four rows a line, written as wide as it is.
mod lines_across_one_value_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(TestedMicrokernel::UnitLinesWrite, Across, 1, 4, Plain, [
        sum: ReduceOperation::Sum,
        mean: ReduceOperation::Mean,
        max: ReduceOperation::Max,
    ]);
}

/// A value a line, written as one cell.
mod lines_along_one_value_one_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(TestedMicrokernel::UnitLinesWrite, Along, 1, 1, Plain, [
        sum: ReduceOperation::Sum,
    ]);
}

/// Rows of one value, a value a line: a partial of one cell, three rows of it.
mod partial_along_one_value_one_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(PARTIAL, Along, 1, 1, Plain, [
        sum: ReduceOperation::Sum,
    ]);
}

/// The write of a region's carrier: four values a line, met into the row's cell.
mod region_along_one_line_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(TestedMicrokernel::UnitRegionWrite, Along, 4, 4, Plain, [
        sum: ReduceOperation::Sum,
        mean: ReduceOperation::Mean,
        max: ReduceOperation::Max,
    ]);
}

/// The write of a region's carrier, a row a lane.
mod region_across_one_value_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(TestedMicrokernel::UnitRegionWrite, Across, 1, 4, Plain, [
        sum: ReduceOperation::Sum,
        mean: ReduceOperation::Mean,
    ]);
}
