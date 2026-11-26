use cubecl::{self as cubecl, CubeType, prelude::*};

/// What the units of a region read of its rows: the carrier of the values the reading microkernel's
/// scope read, met across that scope (a unit's own for a unit microkernel, the cube's for a cube
/// one, in the units its doc names), the values each row holds, and the lanes of a line that hold
/// one row's.
#[derive(CubeType)]
pub struct RowsRead<C: CubeType> {
    pub carrier: C,
    pub value_count: usize,
    #[cube(comptime)]
    pub row_lane_count: usize,
}
