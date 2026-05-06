#![allow(non_snake_case)]

use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::tile::{
    BounceTile, CmmaTile, InterleavedTile, MmaTile, PlaneVecTile, RegisterTile, ScopeMarker,
    SharedTile, TileScope, UnitTile, WhiteboxFragment,
};

/// Public tile type. Wraps a [`TileKind`] storage payload and carries the
/// [`TileScope`] generic via a comptime [`ScopeMarker`] field. The inner
/// [`TileKind`] is crate-private; external callers construct via the
/// `Tile::new_*` constructors below and never destructure.
#[derive(CubeType)]
pub struct Tile<N: Numeric, Sc: TileScope, IO: SliceVisibility> {
    pub(crate) kind: TileKind<N, IO>,
    pub(crate) _scope: ScopeMarker<Sc>,
}

/// Storage variants of a tile. The user-facing wrapper is [`Tile`], which adds
/// a [`TileScope`] type parameter; this enum holds only the runtime/storage
/// payload, with no scope generic. Crate-private — constructors live on
/// [`Tile`] (e.g. `Tile::new_SharedMemory`); internal allocators in
/// `tile/data/*.rs` go through [`Tile::from_kind`].
#[derive(CubeType)]
// The variant constructors live behind the `CubeType`-derived
// `TileKind::new_<Variant>` methods, which the `dead_code` lint can't see
// through.
#[allow(dead_code)]
pub(crate) enum TileKind<N: Numeric, IO: SliceVisibility> {
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
    /// Bundles a cmma fragment, an smem scratch slice, and a `WhiteboxFragment`
    /// view. From the caller's perspective it is a single tile; the smem
    /// round-trip is internal to ops dispatch. Only valid when `Sc = Plane`.
    Bounce(BounceTile<N>),
    /// Sentinel "no source" used by [`crate::tile::Tile::copy_from`] to drive
    /// per-variant zero-init of the destination. Currently produced by
    /// optional-stage flows in `cubek-matmul` (when an accumulator stage is
    /// absent). Prefer [`Tile::init_zero`] when you don't need to thread an
    /// optional source through a generic copy.
    None,
}

#[cube]
impl<N: Numeric, Sc: TileScope, IO: SliceVisibility> Tile<N, Sc, IO> {
    /// Crate-internal: builds a [`Tile`] from a [`TileKind`] payload. Used by
    /// the per-variant `new_*` constructors below and the internal
    /// allocators in `tile/data/*.rs`.
    pub(crate) fn from_kind(kind: TileKind<N, IO>) -> Tile<N, Sc, IO> {
        Tile::<N, Sc, IO> {
            kind,
            _scope: ScopeMarker::<Sc> {
                _phantom: PhantomData,
            },
        }
    }

    /// Wraps a shared-memory tile. Used by stage `tile()` impls to expose a
    /// stage slot as a `Tile` for `copy_from`.
    pub fn new_SharedMemory(t: SharedTile<N, IO>) -> Tile<N, Sc, IO> {
        Self::from_kind(TileKind::new_SharedMemory(t))
    }

    /// Builds a "no source" tile that drives zero-init of the destination
    /// when fed into `copy_from`. Produced by optional-stage flows when the
    /// optional stage is absent; consumers can equivalently call
    /// `dest.init_zero(ident)` directly.
    pub fn new_None() -> Tile<N, Sc, IO> {
        Self::from_kind(TileKind::new_None())
    }
}
