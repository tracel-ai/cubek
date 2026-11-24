//! The cube reduce microkernels alone: a cube of four planes reduces a row into its cells or into a
//! partial, rows shorter than its units and several times longer.

use crate::launch::reduce::TestedMicrokernel;

/// The cube's reduce into a partial. A mean is left out, since only the write divides.
const PARTIAL: TestedMicrokernel = TestedMicrokernel::CubePartial { planes: 4 };

/// The microkernel under test.
const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::CubeStaged { planes: 4 };

/// Rows 37 values long along the reduced dimension, a value a line: fewer values than units.
mod along_37_long_one_wide {
    use super::*;
    every_operation!(MICROKERNEL, Along, 37, 1);
}

/// Rows 4096 values long along the reduced dimension, four values a line.
mod along_4096_long_four_wide {
    use super::*;
    every_operation!(MICROKERNEL, Along, 4096, 4);
}

/// Rows 37 values long across the reduced dimension, a row a line.
mod across_37_long_one_wide {
    use super::*;
    every_operation!(MICROKERNEL, Across, 37, 1);
}

/// Rows 300 values long across the reduced dimension, each lane its own row.
mod across_300_long_four_wide {
    use super::*;
    every_operation!(MICROKERNEL, Across, 300, 4);
}

/// The edge rows, 4096 values long along the reduced dimension, four values a line.
mod edges_along {
    use super::*;
    edge_rows!(MICROKERNEL, Along, 4096, 4);
}

/// The edge rows, 300 values long across the reduced dimension, each lane its own row.
mod edges_across {
    use super::*;
    edge_rows!(MICROKERNEL, Across, 300, 4);
}

/// Rows 300 values long along the reduced dimension, a value a line, each a partial.
mod partial_along_300_long_one_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(PARTIAL, Along, 300, 1, Plain, [
        sum: ReduceOperation::Sum,
        prod: ReduceOperation::Prod,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
        max_abs: ReduceOperation::MaxAbs,
        any: ReduceOperation::Any,
        all: ReduceOperation::All,
    ]);
}

/// Rows 300 values long across the reduced dimension, each lane its own row, four a partial.
mod partial_across_300_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(PARTIAL, Across, 300, 4, Plain, [
        sum: ReduceOperation::Sum,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
    ]);
}

/// The cube's read one line of rows wide, 300 values long along the reduced dimension, four values a
/// line.
mod line_along_300_long_four_wide {
    use super::*;
    every_operation!(TestedMicrokernel::CubeLine { planes: 4 }, Along, 300, 4);
}

/// The cube's read one line of rows wide, 37 values long across the reduced dimension, each lane its
/// own row.
mod line_across_37_long_four_wide {
    use super::*;
    every_operation!(TestedMicrokernel::CubeLine { planes: 4 }, Across, 37, 4);
}

/// The cube's read two lines of rows wide, 300 values long across the reduced dimension, each lane
/// its own row.
mod columns_across_300_long_four_wide {
    use super::*;
    every_operation!(
        TestedMicrokernel::CubeColumns {
            planes: 4,
            columns: 2
        },
        Across,
        300,
        4
    );
}

/// The cube's read into a carrier, rows 300 values long along the reduced dimension, a value a line.
/// A mean is left out, since only the write divides.
mod strided_along_300_long_one_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(TestedMicrokernel::CubeStrided { planes: 4 }, Along, 300, 1, Plain, [
        sum: ReduceOperation::Sum,
        prod: ReduceOperation::Prod,
        max: ReduceOperation::Max,
        min: ReduceOperation::Min,
    ]);
}

/// The same across the reduced dimension, each lane its own row.
mod strided_across_300_long_four_wide {
    use super::*;
    use cubek_microkernel::reduce::ReduceOperation;
    reduce_tests!(TestedMicrokernel::CubeStrided { planes: 4 }, Across, 300, 4, Plain, [
        sum: ReduceOperation::Sum,
        max: ReduceOperation::Max,
    ]);
}
