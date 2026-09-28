//! Who moves an operand's bytes: the [`Delivery`] (the cube's own units, or the TMA engine). What
//! is bound for it to move is the [`Input`](crate::launch::Input) the kernel takes.
//!
//! How the operand is *stored* is a separate fact, riding the spec's [`Storage`](crate::Storage):
//! a storage-tiled operand states the level its tile is the tile of; every mover here serves it.
//! Orthogonal on purpose, so a weight tiled to the stage can move under TMA as well as the units.

use crate::{Refusal, Rendezvous, Space};

/// Who moves an operand into a stage: the cube's own units (a cooperative buffer copy, or a
/// coordinate-backed materialization with no buffer at all), or the TMA engine. Read off a tile
/// via [`delivery`](crate::Tile::delivery); the staging sync comes from it.
///
/// Storage-tiledness is not a variant here. A storage tile is a fact of the data, stated by the
/// spec's [`Storage`](crate::kind::Storage), and it only decides how wide a run each stage is:
/// under `Copy` the units copy that run, under `Tma` it is the box the engine fetches.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    #[default]
    Copy,
    Procedural,
    Tma,
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
            Delivery::Copy | Delivery::Procedural => Ok(()),
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

    /// The synchronization required to materialize this source in a staging slot.
    pub(crate) fn rendezvous(&self) -> Rendezvous {
        match self {
            Delivery::Copy | Delivery::Procedural => Rendezvous::Cube,
            Delivery::Tma => Rendezvous::Barrier,
        }
    }
}
