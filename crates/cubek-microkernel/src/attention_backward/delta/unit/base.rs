use crate::attention::{DV, G, Q};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Levels, Partitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Queries of one head a unit owns whole: their dot product of the output with its gradient, read
/// a line of each at a time along the whole value dim. It asks nothing of the hardware.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDeltaMicrokernel {
    pub settings: UnitDeltaMicrokernelSettings,
    /// The value dim a row of the output spans.
    pub val_dim: NonZeroUsize,
}

/// The queries a unit owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct UnitDeltaMicrokernelSettings {
    pub seq_q: NonZeroUsize,
}

impl UnitDeltaMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        seq_q: NonZeroUsize::MIN,
    };
}

impl core::fmt::Display for UnitDeltaMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "queries={}", self.seq_q)
    }
}

impl UnitDeltaMicrokernel {
    /// The levels inside one unit's queries, which a routine states the rest of the partitioning
    /// around: its queries of one head, the whole value dim.
    pub fn tiling(&self) -> Levels {
        Levels::leaf(&[
            (Q, self.settings.seq_q.get()),
            (G, 1),
            (DV, self.val_dim.get()),
        ])
    }
}

/// Its queries.
impl core::fmt::Display for UnitDeltaMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl Microkernel for UnitDeltaMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = UnitDeltaMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type Partitioning = Partitioning;

    const DECLINE: &'static str = "the unit dot product asks nothing of the hardware";

    /// A query a unit.
    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        vec![Self::Settings::DEFAULT]
    }

    /// One, at the item's value dim; an item with no value dim has nothing to sum.
    fn candidates(
        _hw: &Hardware,
        item: &AttentionBackwardWorkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        let settings = *settings;
        NonZeroUsize::new(item.val_dim)
            .map(|val_dim| Self { settings, val_dim })
            .into_iter()
    }

    /// The output and its gradient are read a line of each at a time, so they share a width; the
    /// dot products are written a value a query.
    fn vectorize(
        &self,
        _partitioning: &Partitioning,
        vectorization: &mut Vectorization<AttentionBackwardOperand>,
    ) {
        use AttentionBackwardOperand as Operand;
        vectorization.require(VectorRequirement::Together {
            operand: Operand::Out,
            with: Operand::OutGradient,
        });
        vectorization.require(VectorRequirement::Elementwise {
            operand: Operand::Delta,
        });
    }

    /// No multiplication: its reads are its routine's to count, which places its units.
    fn estimate(
        &self,
        _item: &AttentionBackwardWorkItem,
        _partitioning: &Partitioning,
        _planes: CubePlanes,
        _estimate: &mut KernelEstimate,
    ) {
    }
}
