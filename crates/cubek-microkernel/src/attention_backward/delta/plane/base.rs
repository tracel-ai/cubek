use crate::attention::{DV, G, Q};
use crate::attention_backward::{AttentionBackwardOperand, AttentionBackwardWorkItem};
use core::num::NonZeroUsize;
use cubecl::tile::launch::{VectorRequirement, Vectorization};
use cubecl::tile::{Levels, Partitioning};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// Queries of one head a plane owns: their dot product of the output with its gradient, its units
/// sharing the queries' rows a line of each at a time along the whole value dim.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneDeltaMicrokernel {
    pub settings: PlaneDeltaMicrokernelSettings,
    /// The value dim a row of the output spans.
    pub val_dim: NonZeroUsize,
}

/// The queries a plane owns.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct PlaneDeltaMicrokernelSettings {
    pub seq_q: NonZeroUsize,
}

impl PlaneDeltaMicrokernelSettings {
    pub const DEFAULT: Self = Self {
        seq_q: NonZeroUsize::new(16).unwrap(),
    };
}

impl core::fmt::Display for PlaneDeltaMicrokernelSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "queries={}", self.seq_q)
    }
}

impl PlaneDeltaMicrokernel {
    /// The levels inside one plane's queries, which a routine states the rest of the partitioning
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
impl core::fmt::Display for PlaneDeltaMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.settings.fmt(f)
    }
}

impl Microkernel for PlaneDeltaMicrokernel {
    type WorkItem = AttentionBackwardWorkItem;
    type Settings = PlaneDeltaMicrokernelSettings;
    type Operand = AttentionBackwardOperand;
    type CubeSpace = Partitioning;

    const DECLINE: &'static str = "the plane dot product runs on a device that commits to one plane \
         width";

    /// Sixteen queries a plane.
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

    /// No multiplication: its reads are its routine's to count, which places its planes.
    fn estimate(
        &self,
        _item: &AttentionBackwardWorkItem,
        _partitioning: &Partitioning,
        _planes: CubePlanes,
        _estimate: &mut KernelEstimate,
    ) {
    }
}
