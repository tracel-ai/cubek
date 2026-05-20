#![allow(non_snake_case)]

use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::tile::{
    BounceTile, CmmaTile, InterleavedTile, MmaTile, PartitionTile, PlaneVecTile, RegisterTile,
    ScopeMarker, SharedTile, StridedStage, TileScope, UnitTile, WhiteboxFragment,
};

/// Public tile type. Wraps a [`TileKind`] storage payload and carries the
/// [`TileScope`] generic both via a comptime [`ScopeMarker`] field *and*
/// through the kind enum (so variants like
/// [`TileKind::Partition`] can keep typed element tiles). The inner
/// [`TileKind`] is crate-private; external callers construct via the
/// `Tile::new_*` constructors below and never destructure.
#[derive(CubeType)]
pub struct Tile<N: Numeric, Sc: TileScope> {
    pub(crate) kind: TileKind<N, Sc>,
    pub(crate) _scope: ScopeMarker<Sc>,
}

/// Storage variants of a tile. The user-facing wrapper is [`Tile`]; this enum
/// holds the runtime/storage payload. Most variants ignore the [`TileScope`]
/// generic (it shows up only on the outer `Tile`'s `ScopeMarker`); the
/// [`Partition`](TileKind::Partition) variant uses it to keep its inner
/// element tiles typed. Crate-private — constructors live on [`Tile`]
/// (e.g. `Tile::new_SharedMemory`); internal allocators in
/// `tile/variants/*.rs` go through [`Tile::from_kind`].
///
/// # The three axes
///
/// Variants encode (some combination of) three orthogonal concerns:
///
/// - **storage** — *where* the bits live: shared memory, hardware-defined
///   register fragments (CMMA / MMA roles), or generic register arrays.
/// - **distribution** — *how* the elements of the tile are spread across the
///   units of a plane: full per-unit copy, exposed plane fragmentation
///   (`WhiteboxFragmentLayout`), or opaque hardware-defined fragmentation
///   (CMMA / MMA).
/// - **compute** — *which* hardware op (or software shape) can act on it as
///   a matmul accelerator: CMMA, MMA, software register matmul, plane-vector
///   inner product, interleaved.
///
/// Variants below are tagged with which axes they pin. A `[fused]` tag
/// flags variants that lock multiple axes at once — these are the natural
/// candidates for the next refactor (split storage from compute, or
/// distribution from compute, so a new backend doesn't require a new
/// enum variant).
#[derive(CubeType)]
// The variant constructors live behind the `CubeType`-derived
// `TileKind::new_<Variant>` methods, which the `dead_code` lint can't see
// through.
#[allow(dead_code)]
pub(crate) enum TileKind<N: Numeric, Sc: TileScope> {
    /// `[storage = smem]`. Pure transport: a stage slot exposed as a tile so
    /// it can be the source / destination of [`Tile::copy_from`]. No
    /// distribution semantics (the caller addresses smem directly), no
    /// compute capability of its own.
    SharedMemory(SharedTile<N>),
    /// `[storage = registers, distribution = opaque-cmma, compute = cmma]
    /// [fused]`. Hardware-defined CMMA fragment. The fragment layout (and
    /// which lane holds which element) is opaque — only the CMMA load/exec/
    /// store path can interact with it. Bundling all three axes is what
    /// makes this variant fragile when adding a new accelerator.
    Cmma(CmmaTile<N>),
    /// `[storage = registers, distribution = opaque-mma, compute = mma]
    /// [fused]`. Hardware-defined MMA fragment, with the operand role
    /// (Lhs / Rhs / Acc) carried by the inner [`crate::tile::variants::MmaFragment`].
    /// Same fused-axes pattern as `Cmma`, plus an extra role axis inside.
    Mma(MmaTile<N>),
    /// `[storage = registers, distribution = per-unit-full, compute =
    /// register-matmul] [fused]`. Software register matmul: each unit holds
    /// a full copy of the tile and computes the product in registers
    /// directly. Convenient label, but the variant locks a per-unit-full
    /// distribution into the matmul kind — these axes should split.
    Register(RegisterTile<N>),
    /// `[storage = registers, distribution = plane-vector, compute =
    /// plane-vec-matmul] [fused]`. Lhs is broadcast to a single vector per
    /// unit; rhs/acc are per-column vectors; the matmul is realized as a
    /// `plane_sum` of element-wise products.
    PlaneVec(PlaneVecTile<N>),
    /// `[storage = registers, distribution = plane-interleaved-on-k,
    /// compute = interleaved-matmul] [fused]`. K dimension is split across
    /// plane units; each unit computes a partial product and the final
    /// reduction is a `plane_sum`.
    Interleaved(InterleavedTile<N>),
    /// `[storage = registers, distribution = per-unit-full, compute = none]`.
    /// Each unit holds a full row-major copy of the tile. Pure
    /// distribution + storage variant — no matmul capability; used by
    /// rowwise / softmax paths that walk the tile element-wise.
    /// Only valid when `Sc = Unit`.
    Unit(UnitTile<N>),
    /// `[storage = registers, distribution = plane-exposed, compute = none]`.
    /// The tile is fragmented across plane units with a layout that is
    /// *visible* to ops (see [`crate::tile::variants::WhiteboxFragmentLayout`]),
    /// in contrast to the opaque CMMA / MMA fragments. Drives the
    /// cross-plane reductions used by softmax. Only valid when
    /// `Sc = Plane`.
    WhiteboxFragment(WhiteboxFragment<N>),
    /// `[storage = registers + smem-scratch, distribution = opaque-cmma
    /// (with whitebox view), compute = cmma]`. Bundles a CMMA fragment,
    /// an smem scratch slice, and a [`WhiteboxFragment`] view of the same
    /// data. Lets attention's softmax run rowwise ops on the whitebox view
    /// while the matmul still uses the cmma fragment; the smem round-trip
    /// is hidden inside `BounceTile` methods. Only valid when `Sc = Plane`.
    Bounce(BounceTile<N>),
    /// `[storage = smem, distribution = stage-level]`. Type-erased view of a
    /// `StridedStageMemory` buffer. Wired in PR 3+ for `.partition()` /
    /// `.mma` dispatch at the stage level. The variant is the "stage is a
    /// tile" hook: a unit-scope `Tile` whose kind is `Stage` represents one
    /// compute primitive's view of the stage; a plane-scope `Tile` whose
    /// kind is `Stage` represents the whole-plane view.
    Stage(StridedStage<N>),
    /// `[fused = sequence-of-instruction-tiles]`. Per-primitive collection
    /// of accumulator tiles (replaces the standalone `Accumulators<MP, Sc>`
    /// wrapper). The element tiles share the partition's [`TileScope`] `Sc`.
    Partition(PartitionTile<N, Sc>),
    /// `[sentinel]`. "No source" used by [`Tile::copy_from`] to drive
    /// per-variant zero-init of the destination. Produced by optional-stage
    /// flows in `cubek-matmul` (when an accumulator stage is absent).
    /// Prefer [`Tile::init_zero`] when you don't need to thread an optional
    /// source through a generic copy.
    None,
}

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc> {
    /// Crate-internal: builds a [`Tile`] from a [`TileKind`] payload. Used by
    /// the per-variant `new_*` constructors below and the internal
    /// allocators in `tile/data/*.rs`.
    pub(crate) fn from_kind(kind: TileKind<N, Sc>) -> Tile<N, Sc> {
        Tile::<N, Sc> {
            kind,
            _scope: ScopeMarker::<Sc> {
                _phantom: PhantomData,
            },
        }
    }

    /// Wraps a shared-memory tile. Used by stage `tile()` impls to expose a
    /// stage slot as a `Tile` for `copy_from`.
    pub fn new_SharedMemory(t: SharedTile<N>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_SharedMemory(t))
    }

    /// Wraps a type-erased stage view as a tile. The `Sc` generic on `Tile`
    /// reflects which compute primitive's view this is (e.g. one unit's
    /// slice vs the whole plane's slice).
    pub fn new_Stage(t: StridedStage<N>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_Stage(t))
    }

    /// Wraps a partition of accumulator tiles. The element tiles share the
    /// partition's `Sc`.
    pub fn new_Partition(t: PartitionTile<N, Sc>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_Partition(t))
    }

    /// Builds a "no source" tile that drives zero-init of the destination
    /// when fed into `copy_from`. Produced by optional-stage flows when the
    /// optional stage is absent; consumers can equivalently call
    /// `dest.init_zero(ident)` directly.
    pub fn new_None() -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_None())
    }
}
