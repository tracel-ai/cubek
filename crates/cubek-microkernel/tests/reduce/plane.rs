//! The plane reduce microkernel alone: a plane reduces a row, rows shorter and longer than its
//! lanes.

use crate::launch::reduce::TestedMicrokernel;

const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::PlaneShuffle;

/// The plane's rows 37 values long along the reduced dimension, a value a line.
mod along_37_long_one_wide {
    use super::*;
    every_operation!(MICROKERNEL, Along, 37, 1);
}

/// Rows 20 values long along the reduced dimension, four values a line: fewer lines than lanes.
mod along_20_long_four_wide {
    use super::*;
    every_operation!(MICROKERNEL, Along, 20, 4);
}

/// Rows 1024 values long along the reduced dimension, four values a line: many lines a lane.
mod along_1024_long_four_wide {
    use super::*;
    every_operation!(MICROKERNEL, Along, 1024, 4);
}

/// The plane's rows 37 values long across the reduced dimension, a row a line.
mod across_37_long_one_wide {
    use super::*;
    every_operation!(MICROKERNEL, Across, 37, 1);
}

/// The plane's rows 300 values long across the reduced dimension, each lane its own row.
mod across_300_long_four_wide {
    use super::*;
    every_operation!(MICROKERNEL, Across, 300, 4);
}

/// The plane's edge rows, 1024 values long along the reduced dimension, four values a line.
mod edges_along {
    use super::*;
    edge_rows!(MICROKERNEL, Along, 1024, 4);
}

/// The plane's edge rows, 300 values long across the reduced dimension, each lane its own row.
mod edges_across {
    use super::*;
    edge_rows!(MICROKERNEL, Across, 300, 4);
}

/// The plane's read one line of rows wide, 300 values long along the reduced dimension, four values
/// a line.
mod line_along_300_long_four_wide {
    use super::*;
    every_operation!(TestedMicrokernel::PlaneLine, Along, 300, 4);
}

/// The plane's read one line of rows wide, 37 values long across the reduced dimension, each lane
/// its own row.
mod line_across_37_long_four_wide {
    use super::*;
    every_operation!(TestedMicrokernel::PlaneLine, Across, 37, 4);
}

/// The plane's read two lines of rows wide, 300 values long across the reduced dimension, each lane
/// its own row.
mod columns_across_300_long_four_wide {
    use super::*;
    every_operation!(
        TestedMicrokernel::PlaneColumns { columns: 2 },
        Across,
        300,
        4
    );
}
