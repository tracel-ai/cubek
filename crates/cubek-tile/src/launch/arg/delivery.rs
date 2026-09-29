//! Who moves an operand's bytes, and whether they have landed when the fill returns: the
//! [`Delivery`]. What is bound for it to move is the [`Input`](crate::launch::Input) the kernel
//! takes.
//!
//! How the operand is *stored* is a separate fact, riding the spec's [`Storage`](crate::Storage):
//! a storage-tiled operand states the level its tile is the tile of; every mover here serves it.
//! Orthogonal on purpose, so a weight tiled to the stage can move under TMA as well as the units.

use crate::{Refusal, Rendezvous, Space};

/// Who moves an operand into a stage, and whether its bytes have landed when the fill returns. One
/// variant per mechanism, each named for both. Read off a tile via
/// [`delivery`](crate::Tile::delivery); the staging sync comes from it.
///
/// A tensor is moved by [`SyncPerUnit`](Delivery::SyncPerUnit) unless its spec states
/// [`AsyncPerUnit`](Delivery::AsyncPerUnit) or [`AsyncBulk`](Delivery::AsyncBulk)
/// ([`TileSpec::delivery`](crate::TileSpec::delivery)); a tensor map by [`Tma`](Delivery::Tma), and
/// a coordinate-backed tile by [`Procedural`](Delivery::Procedural).
///
/// Storage-tiledness is not a variant here. A storage tile is a fact of the data, stated by the
/// spec's [`Storage`](crate::kind::Storage), and it only decides how wide a run each stage is:
/// under a per-unit delivery the units copy that run, under `Tma` it is the box the engine fetches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    /// Every unit loads its own lines and stores them, through its registers: landed when the fill
    /// returns, and a line can be decoded on its way.
    #[default]
    SyncPerUnit,
    /// Every unit hands its own lines to the copy engine (`cp.async`), memory to memory: they land
    /// after the fill returns, as they lie, and something the slot waits on tracks them.
    AsyncPerUnit,
    /// One unit hands the whole stage to the copy engine as one contiguous run
    /// (`cp.async.bulk`): it lands after the fill returns, counted in bytes on the slot's barrier.
    AsyncBulk,
    /// One unit hands the stage's box to the TMA engine through a tensor map
    /// (`cp.async.bulk.tensor`): it lands after the fill returns, counted in bytes on the slot's
    /// barrier.
    Tma,
    /// Every unit computes its values from their coordinates: nothing is read, and the stage
    /// holds them when the fill returns.
    Procedural,
}

/// CUDA caps each TMA box dimension at 256; a bulk copy fills one smem stage, so the
/// stage edges are the box dims ([`Delivery::moves`]).
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    /// Whether this delivery moves `stage`, one transfer a stage, and why not where it cannot.
    ///
    /// Only [`Tma`](Delivery::Tma) is bounded: its bulk copy fills a stage per box, so every stage
    /// edge is a box edge, and the descriptor caps each. The cube's own units, copying or
    /// computing, move any stage. A selector sizing a stage asks this of each candidate rather than
    /// holding the limit itself.
    pub fn moves(self, stage: &Space) -> Result<(), Refusal> {
        match self {
            Delivery::SyncPerUnit
            | Delivery::AsyncPerUnit
            | Delivery::AsyncBulk
            | Delivery::Procedural => Ok(()),
            Delivery::Tma => match stage
                .extents()
                .into_iter()
                .find(|&(_, edge)| edge > TMA_MAX_BOX_DIM)
            {
                Some((axis, edge)) => Err(Refusal::BoxPastDescriptor {
                    axis,
                    edge,
                    most: TMA_MAX_BOX_DIM,
                }),
                None => Ok(()),
            },
        }
    }

    pub fn is_tma(&self) -> bool {
        matches!(self, Delivery::Tma)
    }

    /// Whether the fill returns before its bytes land.
    pub fn is_async(&self) -> bool {
        match self {
            Delivery::SyncPerUnit | Delivery::Procedural => false,
            Delivery::AsyncPerUnit | Delivery::AsyncBulk | Delivery::Tma => true,
        }
    }

    /// The synchronization required to materialize this source in a staging slot. A bulk copy
    /// completes only on an mbarrier counting its bytes. A per-unit async copy could also wait on
    /// its units' own copy groups; the slot tracks it on the same mbarrier for now.
    pub(crate) fn rendezvous(&self) -> Rendezvous {
        if self.is_async() {
            Rendezvous::Barrier
        } else {
            Rendezvous::Cube
        }
    }

    /// Whether the copy writes through the async proxy and completes on the slot's barrier as
    /// transaction bytes (`cp.async.bulk`, TMA), so the barrier's initialization must be fenced
    /// into that proxy before the first copy. A per-unit `cp.async` stays in the generic proxy.
    pub(crate) fn through_async_proxy(&self) -> bool {
        matches!(self, Delivery::AsyncBulk | Delivery::Tma)
    }

    /// Whether every unit of the cube takes part in the fill, rather than one elected issuer.
    pub(crate) fn every_unit_fills(&self) -> bool {
        match self {
            Delivery::SyncPerUnit | Delivery::AsyncPerUnit | Delivery::Procedural => true,
            Delivery::AsyncBulk | Delivery::Tma => false,
        }
    }
}
