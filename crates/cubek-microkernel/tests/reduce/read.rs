//! The strided read the plane and cube microkernels share, alone: a unit reads every third line
//! of its row from the first into a carrier of values. The operations are those whose carrier is
//! the values themselves; a mean is left out, since only the write divides.

use crate::launch::reduce::TestedMicrokernel;

/// The microkernel under test.
const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::UnitStridedReduce { stride: 3 };

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
