//! Where a unit stands in its cube, as the microkernels that write once a cube read it, and where
//! its plane stages a partial for the cube's planes to meet.

use cubecl::{self as cubecl, cube, prelude::*};

/// Whether this unit is its cube's first: the one that writes what the whole cube reduced.
#[cube]
pub(crate) fn is_first_unit_of_cube() -> bool {
    PLANE_POS == 0 && UNIT_POS_PLANE == 0
}

/// The slot of shared memory `plane` stages column `column`'s partial in, of the `columns` each
/// plane holds: a plane's slots side by side, the planes one after another.
#[cube]
pub(crate) fn stage_slot(plane: usize, column: usize, #[comptime] columns: usize) -> usize {
    plane * columns + column
}
