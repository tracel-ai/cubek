//! The reads other microkernels share, alone. The operations are those whose carrier is the values
//! themselves.

use crate::launch::reduce::TestedMicrokernel;

/// The strided read: a unit reads every third line of its row from the first. It divides nothing, so
/// no mean.
const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::UnitStridedReduce { stride: 3 };

/// The walked read: a plane's units lie across two lines of rows, so each column's first unit reads
/// every value as many apart as the column holds units.
const WALK: TestedMicrokernel = TestedMicrokernel::UnitWalkReduce { columns: 2 };

/// The reduce of stored partials.
const PARTIALS: TestedMicrokernel = TestedMicrokernel::UnitPartialsReduce;

/// Rows 37 values long along the reduced dimension, a value a line: twelve full steps, then the
/// last line, 36, read past them.
mod along_37_long_one_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(MICROKERNEL, Along, 37, 1, Plain, [
        sum: ReduceOperation::Sum,
        prod: ReduceOperation::Prod,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
        max_abs: ReduceOperation::MaxAbs,
        any: ReduceOperation::Any,
        all: ReduceOperation::All,
    ]);
}

/// Rows 36 values long along the reduced dimension: twelve full steps and no line past them.
mod along_36_long_one_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(MICROKERNEL, Along, 36, 1, Plain, [
        sum: ReduceOperation::Sum,
    ]);
}

/// Rows 37 values long across the reduced dimension, each lane its own row.
mod across_37_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(MICROKERNEL, Across, 37, 4, Plain, [
        sum: ReduceOperation::Sum,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
    ]);
}

/// Walked across rows 300 values long, each lane its own row: the last step of the walk is cut
/// short.
mod walk_across_300_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(WALK, Across, 300, 4, Plain, [
        sum: ReduceOperation::Sum,
        prod: ReduceOperation::Prod,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
        max_abs: ReduceOperation::MaxAbs,
        any: ReduceOperation::Any,
        all: ReduceOperation::All,
    ]);
}

/// Walked across rows 320 values long, no step of the walk cut short whatever the plane's width.
mod walk_across_320_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(WALK, Across, 320, 4, Plain, [
        sum: ReduceOperation::Sum,
    ]);
}

/// Nine partials of four lanes of one row each.
mod partials_along_36_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(PARTIALS, Along, 36, 4, Plain, [
        sum: ReduceOperation::Sum,
        prod: ReduceOperation::Prod,
        mean: ReduceOperation::Mean,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
    ]);
}

/// Thirty-seven partials, each lane its own row.
mod partials_across_37_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(PARTIALS, Across, 37, 4, Plain, [
        sum: ReduceOperation::Sum,
        mean: ReduceOperation::Mean,
        max: ReduceOperation::Max,
    ]);
}
