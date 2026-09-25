//! A unit's place in the team it shares a leaf with, which the attention
//! matmul leaves and the unit-owned softmax both distribute a tile by.

use cubecl::prelude::*;

/// A unit's place in the team it shares a leaf with: its index among the team's units, and how
/// many there are.
///
/// The leaves that distribution one tile to a team — a unit's cyclic share of the attention columns, a
/// unit's run of softmax rows — take this rather than reading the cube's x dim, because a team
/// is the caller's cut and not a dimension of the cube: a team wider than a plane spans several
/// rows of it, and a cube read off a partitioning is a plane wide whatever the team is. A kernel
/// whose levels distribute the team reads the two off them; [`along_x`](TeamUnit::along_x) is the
/// team a kernel lays on the cube's x dim by hand.
///
/// Its expand type is `Clone`, so a cube struct that holds one — a fold's per-team view, a
/// [`RowState`](crate::RowState) — can hand a copy on rather than rebuild it from its fields.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct TeamUnit {
    /// This unit, among the team's.
    pub index: usize,
    /// Units the team holds.
    pub units: usize,
}

#[cube]
impl TeamUnit {
    /// This unit in a team of `units`, at `index`.
    pub fn new(index: usize, units: usize) -> TeamUnit {
        TeamUnit { index, units }
    }

    /// A team laid along the cube's x dim: the whole cube where its y dim is one, one split
    /// team per row of it otherwise.
    pub fn along_x() -> TeamUnit {
        TeamUnit {
            index: UNIT_POS_X as usize,
            units: CUBE_DIM_X as usize,
        }
    }
}
