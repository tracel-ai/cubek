use cubecl::prelude::*;

use crate::tile::{
    BounceTile, CmmaTile, InterleavedTile, MmaTile, PlaneVecTile, RegisterTile, ScopeMarker,
    SharedTile, TileScope, UnitTile, WhiteboxFragment,
};

#[derive(CubeType)]
pub enum Tile<N: Numeric, Sc: TileScope, IO: SliceVisibility> {
    SharedMemory(SharedTile<N, IO>),
    Cmma(CmmaTile<N>),
    Mma(MmaTile<N>),
    Register(RegisterTile<N>),
    PlaneVec(PlaneVecTile<N>),
    Interleaved(InterleavedTile<N>),
    /// Each unit holds a full row-major copy of the tile in registers.
    /// Only valid when `Sc = Unit`.
    Unit(UnitTile<N>),
    /// The tile is fragmented across plane units, with the layout exposed.
    /// Only valid when `Sc = Plane`.
    WhiteboxFragment(WhiteboxFragment<N>),
    /// Bundles a cmma fragment, an smem scratch slice, and a `WhiteboxFragment` view.
    /// From the caller's perspective it is a single tile; the smem round-trip
    /// is internal to ops dispatch. Only valid when `Sc = Plane`.
    Bounce(BounceTile<N>),
    Broadcasted(Value<N>),
    None,
    _Phantom(ScopeMarker<Sc>),
}

/// Wrapper over val to make enum work
#[derive(CubeType)]
pub struct Value<E: Numeric> {
    pub val: E,
}
