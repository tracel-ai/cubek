use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Where a walk reads the state it resumes from and writes the states it keeps: the one thing a
/// walk over states it owns and one in a store's slots differ in.
///
/// In a slot, `read` and `write` are one tile, the row's slot, and the state after the last
/// position is the one written back. A row that has walked no position is `fresh`: it starts from
/// zeros and never reads its slot, which holds whatever it held before.
#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct StateAccess {
    pub read: Tile,
    pub write: Tile,
    pub fresh: bool,
    pub first_kept: usize,
}

#[cube]
impl StateAccess {
    /// The walk over `state`, keeping the states from position `first_kept` on in `kept`; no row
    /// is fresh.
    pub fn owned(state: &Tile, kept: &Tile, first_kept: usize) -> StateAccess {
        StateAccess {
            read: state.clone(),
            write: kept.clone(),
            fresh: false,
            first_kept,
        }
    }

    /// The walk over `slot`, the row's tile of its store, fresh where its `positions` tile holds
    /// zero; the state after position `first_kept` is written back.
    pub fn in_slot(slot: &Tile, positions: &Tile, first_kept: usize) -> StateAccess {
        let walked = positions
            .flat::<u32, Const<1>>()
            .read(0usize)
            .extract(0usize);
        StateAccess {
            read: slot.clone(),
            write: slot.clone(),
            fresh: walked == 0u32,
            first_kept,
        }
    }
}
