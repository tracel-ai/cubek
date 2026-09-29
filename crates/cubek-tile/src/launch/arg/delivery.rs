//! Who moves an operand's bytes into a stage: the [`Delivery`].

use crate::{Refusal, Rendezvous, Space};

/// Who moves an operand into a stage: the cube's own units, or the TMA engine.
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, Default)]
pub enum Delivery {
    #[default]
    Copy,
    Procedural,
    Tma,
}

/// CUDA's cap on each TMA box dimension.
const TMA_MAX_BOX_DIM: usize = 256;

impl Delivery {
    /// Whether this delivery moves `stage` in one transfer, and why not where it cannot.
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
