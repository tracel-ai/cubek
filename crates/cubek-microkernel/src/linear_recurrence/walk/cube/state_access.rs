use crate::linear_recurrence::walked;
use cubecl::tile::Tile;
use cubecl::{self as cubecl, CubeType, cube, prelude::*};

/// Where a walk reads the state it resumes from and writes the states it keeps: the one thing a
/// walk over states it owns and one in a store's slots differ in.
///
/// In a slot, `read` and `write` are one tile, the row's slot, and the state after the last
/// position is the one written back. A row of length zero is `fresh`: it starts from zeros and
/// never reads its slot, which holds whatever it held before.
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
    /// The walk over `state`, keeping the states from position `first_kept` on in `kept` where
    /// there is one and writing the state after that position back to `state` otherwise; fresh
    /// where the `lengths` tile, if any, holds zero.
    pub fn new(
        state: &Tile,
        kept: &ComptimeOption<Tile>,
        lengths: &ComptimeOption<Tile>,
        first_kept: usize,
    ) -> StateAccess {
        StateAccess {
            read: state.clone(),
            write: kept.clone().unwrap_or(state.clone()),
            fresh: is_fresh(lengths),
            first_kept,
        }
    }
}

/// Whether the row of the `lengths` tile, if any, has walked no position.
#[cube]
fn is_fresh(lengths: &ComptimeOption<Tile>) -> bool {
    #[comptime]
    match lengths {
        ComptimeOption::Some(lengths) => walked(lengths) == 0u32,
        ComptimeOption::None => false,
    }
}
