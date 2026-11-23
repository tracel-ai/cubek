//! The cube reduce microkernel alone: a cube of four planes reduces a row, rows shorter than its
//! units and several times longer.

use crate::launch::reduce::TestedMicrokernel;

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
