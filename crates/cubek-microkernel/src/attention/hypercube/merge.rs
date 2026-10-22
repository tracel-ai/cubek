use crate::attention::AttentionSlotShape;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// How the slots of a box are merged into its queries of the output, by
/// [`plane_merge_attention_microkernel()`](fn@crate::attention::merge::plane::plane_merge_attention_microkernel)
/// either way.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HypercubeMerge {
    /// In the same launch, by the last of the box's cubes to count itself in: one launch, but one
    /// plane merges a whole box while the rest of the device has nothing left to do. Needs a
    /// device where one cube's stores reach another within a launch.
    LastArrival,
    /// By a second launch, one plane a row: every row is merged at once, and no cube counts
    /// itself in anywhere.
    SecondLaunch,
}

/// What a merge on the last arrival declines on a device where one cube's stores do not reach
/// another within a launch: the last cube of a box reads what the others stored.
pub const HYPERCUBE_CUBES_UNMET: Refusal = Refusal::Declined {
    why: "merging on the last arrival needs a device where one cube's stores reach another \
              within a launch",
};

impl HypercubeMerge {
    /// Whether `hw` can merge this way.
    ///
    /// # Errors
    ///
    /// [`HYPERCUBE_CUBES_UNMET`] for the last arrival where one cube's stores do not reach another
    /// within a launch.
    pub fn admits(&self, hw: &Hardware) -> Result<(), Refusal> {
        match self {
            Self::LastArrival if !hw.properties.features.device_memory_scope => {
                Err(HYPERCUBE_CUBES_UNMET)
            }
            Self::LastArrival | Self::SecondLaunch => Ok(()),
        }
    }

    /// Bytes of shared memory a cube holds to merge this way: the word a cube learns it came last
    /// in, for the last arrival.
    pub fn shared_memory(&self) -> usize {
        match self {
            Self::LastArrival => size_of::<u32>(),
            Self::SecondLaunch => 0,
        }
    }
}

impl core::fmt::Display for HypercubeMerge {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::LastArrival => "last-arrival",
            Self::SecondLaunch => "second-launch",
        })
    }
}

/// What a merge of slots is compiled for: the slots of every query, the cubes sharing a box, which
/// count themselves in, and the queries a merging plane holds
/// ([`merge_shape`](crate::attention::hypercube::HypercubeAttentionMicrokernel::merge_shape)).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct HypercubeMergeShape {
    pub slots: AttentionSlotShape,
    pub cubes: usize,
    pub seq_q: usize,
}
