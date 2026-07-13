//! How an operand's bytes move: the [`Delivery`] (strided cooperative copy or TMA bulk
//! copy) and its type-level twin [`DeliveryFamily`], which lets one kernel body serve
//! both argument types.

use cubecl::prelude::*;

use crate::{Space, StridedTileArg, Tile, TmaTileArg};

/// How an operand's bytes move out of it: a strided cooperative copy or a TMA hardware
/// bulk copy. Read off a tile via [`delivery`](crate::Tile::delivery); the staging sync
/// comes from it.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    #[default]
    Strided,
    Tma,
}

/// CUDA caps each TMA box dimension at 256; a bulk copy fills one smem stage, so the
/// stage edges are the box dims.
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// Reject a plan the TMA descriptor path can't encode, so a bad plan fails here as a
    /// clean error instead of at descriptor encoding on the driver. `boxes` are the
    /// bulk-copy box dims (one stage per box); `batched` = any surviving batch dim.
    /// A no-op unless this is [`Delivery::Tma`].
    pub fn validate_tma(&self, boxes: &[usize], batched: bool) -> Result<(), String> {
        if !self.is_tma() {
            return Ok(());
        }
        // The descriptor is 3-D `(batch, row, col)`; surviving batch dims need a
        // batch-aware descriptor path not wired yet.
        if batched {
            return Err("TMA: batched problems are not supported yet".to_string());
        }
        if let Some(&max) = boxes.iter().max()
            && max > TMA_MAX_BOX_DIM
        {
            return Err(format!(
                "TMA: box {boxes:?} exceeds the {TMA_MAX_BOX_DIM}-per-axis box limit"
            ));
        }
        Ok(())
    }
}

/// How a derived smem stage lays out its buffer: storage-tiled at the final tile (one
/// contiguous block per fragment) or plain strided rows (legacy `sync_full_strided`).
/// A per-operand comptime plan config ([`stage`](crate::StridedTileSource::stage)).
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

/// [`Delivery::Strided`]'s family: a plain tensor ([`StridedTileArg`]), cooperatively copied.
pub struct Strided;

/// [`Delivery::Tma`]'s family: a tensor map ([`TmaTileArg`]), hardware bulk-copied.
pub struct Tma;

#[cube]
impl DeliveryFamily for Strided {
    type Arg<E: Numeric> = StridedTileArg<'static, E>;

    fn tile<E: Numeric>(arg: &Self::Arg<E>) -> Tile<E> {
        arg.tile()
    }
}

#[cube]
impl DeliveryFamily for Tma {
    type Arg<E: Numeric> = TmaTileArg<E>;

    fn tile<E: Numeric>(arg: &Self::Arg<E>) -> Tile<E> {
        arg.tile()
    }
}
