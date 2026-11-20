//! Where a unit stands in its cube, as the microkernels that write once a cube read it.

use cubecl::{self as cubecl, cube, prelude::*};

/// Whether this unit is its cube's first: the one that writes what the whole cube reduced.
#[cube]
pub(crate) fn is_first_unit_of_cube() -> bool {
    PLANE_POS == 0 && UNIT_POS_PLANE == 0
}
