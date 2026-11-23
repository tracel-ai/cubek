//! The hypercube reduce microkernel alone: cubes of four planes each reduce a part of the row, the
//! last to finish merging the parts.

use crate::launch::reduce::TestedMicrokernel;

/// Rows 64 values long along the reduced dimension, a value a line, in two parts each shorter than
/// a cube's units.
mod along_two_short_parts {
    use super::*;

    /// The microkernel under test.
    const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::HypercubeLastArrival {
        planes: 4,
        parts: 2,
    };

    every_operation!(MICROKERNEL, Along, 64, 1);
}

/// Rows 8192 values long along the reduced dimension, four values a line, in eight parts each
/// several times a cube's units.
mod along_eight_long_parts {
    use super::*;

    /// The microkernel under test.
    const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::HypercubeLastArrival {
        planes: 4,
        parts: 8,
    };

    every_operation!(MICROKERNEL, Along, 8192, 4);

    /// The edge rows, in the same eight parts.
    mod edges {
        use super::*;
        edge_rows!(MICROKERNEL, Along, 8192, 4);
    }
}

/// Rows 300 values long across the reduced dimension, each lane its own row, in three parts.
mod across_three_parts {
    use super::*;

    /// The microkernel under test.
    const MICROKERNEL: TestedMicrokernel = TestedMicrokernel::HypercubeLastArrival {
        planes: 4,
        parts: 3,
    };

    every_operation!(MICROKERNEL, Across, 300, 4);

    /// The edge rows, in the same three parts.
    mod edges {
        use super::*;
        edge_rows!(MICROKERNEL, Across, 300, 4);
    }
}
