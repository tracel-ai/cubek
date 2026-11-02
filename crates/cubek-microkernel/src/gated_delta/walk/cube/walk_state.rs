use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Where a walk reads the state it resumes from and writes the states it keeps, the one thing
/// that differs between a walk over states it owns and one in a store's slots.
///
/// Owned, `read` is the state tensor, `write` the kept states, and no row is fresh. In a slot,
/// `read` and `write` are one tile placed by the row's slot, the last position's state is the one
/// written, and a row that has walked no position yet is fresh: its slot holds whatever it held
/// before, which the walk never reads.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct WalkState {
    pub read: Tile,
    pub write: Tile,
    pub fresh: bool,
    pub first_kept: usize,
}

#[cube]
impl WalkState {
    /// The walk over `state`, keeping the states from position `first_kept` on in `kept`.
    pub fn owned(state: &Tile, kept: &Tile, first_kept: usize) -> WalkState {
        WalkState {
            read: state.clone(),
            write: kept.clone(),
            fresh: false,
            first_kept,
        }
    }

    /// The walk over `slot`, the row's tile of its store, fresh where its `positions` tile holds
    /// zero; the state after position `first_kept` is written back.
    pub fn in_slot(slot: &Tile, positions: &Tile, first_kept: usize) -> WalkState {
        let walked = positions
            .flat::<u32, Const<1>>()
            .read(0usize)
            .extract(0usize);
        WalkState {
            read: slot.clone(),
            write: slot.clone(),
            fresh: walked == 0u32,
            first_kept,
        }
    }
}
