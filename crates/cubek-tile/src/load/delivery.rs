//! How an operand's bytes move: the [`Delivery`] (strided cooperative copy or TMA bulk
//! copy) and its type-level twin [`DeliveryFamily`], which lets one kernel body serve
//! both argument types.

use cubecl::prelude::*;

use crate::{Space, Tile, TileArg, TmaArg};

/// How an operand's bytes move out of it: a strided cooperative copy or a TMA hardware
/// bulk copy. Read off a tile via [`delivery`](crate::Tile::delivery); the staging sync
/// comes from it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum Delivery {
    Strided,
    Tma,
}

impl Delivery {
    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }
}

/// How a derived smem stage lays out its buffer: storage-tiled at the final tile (one
/// contiguous block per fragment) or plain strided rows (legacy `sync_full_strided`).
/// A per-operand comptime plan config ([`stage`](crate::TileSource::stage)).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub enum StageStorage {
    Tiled,
    Strided,
}

impl StageStorage {
    /// The safe default: a cmma leaf reads whole fragments, so tile its stages; anything
    /// else keeps plain strided rows.
    pub fn for_space(space: &Space) -> Self {
        if space.partitioner().leaf().is_cmma() {
            StageStorage::Tiled
        } else {
            StageStorage::Strided
        }
    }
}

/// [`Delivery`]'s type-level twin: which launchable argument carries an operand and how a
/// kernel serves that argument as a [`Tile`]. A kernel body written over
/// `D: DeliveryFamily` runs strided or TMA unchanged; the launch entry picks the family.
/// One family covers both operands, since [`Sync::of`](crate::Sync::of) rejects a mixed
/// pair anyway.
#[cube]
pub trait DeliveryFamily: Send + core::marker::Sync + 'static {
    /// The launchable argument carrying one operand.
    type Arg<E: Numeric>: LaunchArg + CubeType;

    /// Serve the argument as a [`Tile`].
    fn tile<E: Numeric>(arg: &Self::Arg<E>) -> Tile<E>;
}

/// [`Delivery::Strided`]'s family: a plain tensor ([`TileArg`]), cooperatively copied.
pub struct Strided;

/// [`Delivery::Tma`]'s family: a tensor map ([`TmaArg`]), hardware bulk-copied.
pub struct Tma;

#[cube]
impl DeliveryFamily for Strided {
    type Arg<E: Numeric> = TileArg<'static, E>;

    fn tile<E: Numeric>(arg: &Self::Arg<E>) -> Tile<E> {
        arg.tile()
    }
}

#[cube]
impl DeliveryFamily for Tma {
    type Arg<E: Numeric> = TmaArg<E>;

    fn tile<E: Numeric>(arg: &Self::Arg<E>) -> Tile<E> {
        arg.tile()
    }
}
