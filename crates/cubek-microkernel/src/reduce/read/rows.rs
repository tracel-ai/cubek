use cubecl::{self as cubecl, CubeType, prelude::*};

/// What the units of a region read of its rows: the carrier each holds, met or not, the values each row
/// holds, and the lanes of a line that hold one row's.
#[derive(CubeType)]
pub struct RowsRead<C: CubeType> {
    pub carrier: C,
    pub values: usize,
    #[cube(comptime)]
    pub row_lanes: usize,
}
