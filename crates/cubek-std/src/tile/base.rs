#![allow(non_snake_case)]

use std::marker::PhantomData;

use cubecl::prelude::*;

use crate::tile::{
    BounceTile, CmmaTile, InterleavedTile, MmaTile, PartitionTile, PipelinedTile, PlaneVecTile,
    RegisterTile, RowWise, ScopeMarker, SharedTile, StageTile, TileScope, UnitTile,
    WhiteboxFragment, variants::stage::partition::partition_get_at_mut,
};

/// Public tile type. Wraps a [`TileKind`] payload; the inner enum is
/// crate-private and callers construct via `Tile::new_*`.
#[derive(CubeType)]
pub struct Tile<N: Numeric, Sc: TileScope> {
    pub(crate) kind: TileKind<N, Sc>,
    pub(crate) _scope: ScopeMarker<Sc>,
}

/// Storage variants of a tile.
#[derive(CubeType)]
#[allow(dead_code)]
pub(crate) enum TileKind<N: Numeric, Sc: TileScope> {
    /// Whole-stage view, used for partition-level dispatch.
    Stage(StageTile<N>),
    /// Sequence of per-tile accumulators.
    Partition(PartitionTile<N, Sc>),
    /// Stage slot exposed as a tile (no distribution, no compute).
    SharedTile(SharedTile<N>),

    /// CMMA fragment.
    Cmma(CmmaTile<N>),
    /// MMA fragment; operand role (Lhs/Rhs/Acc) carried inside.
    Mma(MmaTile<N>),
    /// Register-resident tile for the software register matmul.
    Register(RegisterTile<N>),
    /// Plane-vector matmul tile.
    PlaneVec(PlaneVecTile<N>),
    /// Plane-interleaved-on-k matmul tile.
    Interleaved(InterleavedTile<N>),
    /// Per-unit register array. `Sc = Unit`.
    Unit(UnitTile<N>),
    /// Plane-exposed fragment with a visible layout. `Sc = Plane`.
    WhiteboxFragment(WhiteboxFragment<N>),
    /// Per-row vector tile (softmax max/sum state). `Sc = Plane`.
    RowWise(RowWise<N>),

    /// Rhs fragments for the partition matmul (1 = single-buffered, 2 = double).
    Pipelined(PipelinedTile<N, Sc>),
    /// CMMA fragment + smem scratch + whitebox view. `Sc = Plane`.
    Bounce(BounceTile<N>),

    /// Sentinel for zero-init via `copy_from`.
    None,
}

#[cube]
impl<N: Numeric, Sc: TileScope> Tile<N, Sc> {
    pub(crate) fn from_kind(kind: TileKind<N, Sc>) -> Tile<N, Sc> {
        Tile::<N, Sc> {
            kind,
            _scope: ScopeMarker::<Sc> {
                _phantom: PhantomData,
            },
        }
    }

    pub fn new_SharedTile(t: SharedTile<N>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_SharedTile(t))
    }

    pub fn new_Stage(t: StageTile<N>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_Stage(t))
    }

    pub fn new_Partition(t: PartitionTile<N, Sc>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_Partition(t))
    }

    pub fn new_Pipelined(t: PipelinedTile<N, Sc>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_Pipelined(t))
    }

    pub fn new_None() -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_None())
    }

    pub fn new_RowWise(t: RowWise<N>) -> Tile<N, Sc> {
        Self::from_kind(TileKind::new_RowWise(t))
    }

    /// Mutable reference to the `(m, n)` element of a `Partition` tile.
    pub fn partition_tile_at_mut(
        &mut self,
        #[comptime] m: usize,
        #[comptime] n: usize,
        #[comptime] n_cols: usize,
    ) -> &mut Tile<N, Sc> {
        match &mut self.kind {
            TileKind::Partition(p) => partition_get_at_mut::<N, Sc>(p, m, n, n_cols),
            TileKind::SharedTile(_)
            | TileKind::Cmma(_)
            | TileKind::Mma(_)
            | TileKind::Register(_)
            | TileKind::PlaneVec(_)
            | TileKind::Interleaved(_)
            | TileKind::Unit(_)
            | TileKind::WhiteboxFragment(_)
            | TileKind::RowWise(_)
            | TileKind::Bounce(_)
            | TileKind::Stage(_)
            | TileKind::Pipelined(_)
            | TileKind::None => {
                panic!("Tile::partition_tile_at_mut: self.kind is not Partition")
            }
        }
    }
}
