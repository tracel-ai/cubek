//! The unit reduce microkernel alone: a row a unit.

use crate::launch::reduce::TestedMicrokernel;

const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::UnitDirect;

/// The unit's rows 37 values long along the reduced dimension, a value a line.
mod along_37_long_one_wide {
    use super::*;
    every_operation!(MICROKERNEL, Along, 37, 1);
}

/// Rows 64 values long along the reduced dimension, four values a line.
mod along_64_long_four_wide {
    use super::*;
    every_operation!(MICROKERNEL, Along, 64, 4);
}

/// The unit's rows 37 values long across the reduced dimension, a row a line.
mod across_37_long_one_wide {
    use super::*;
    every_operation!(MICROKERNEL, Across, 37, 1);
}

/// The unit's rows 300 values long across the reduced dimension, each lane its own row.
mod across_300_long_four_wide {
    use super::*;
    every_operation!(MICROKERNEL, Across, 300, 4);
}

/// The unit's edge rows, 64 values long along the reduced dimension, four values a line.
mod edges_along {
    use super::*;
    edge_rows!(MICROKERNEL, Along, 64, 4);
}

/// The unit's edge rows, 300 values long across the reduced dimension, each lane its own row.
mod edges_across {
    use super::*;
    edge_rows!(MICROKERNEL, Across, 300, 4);
}
