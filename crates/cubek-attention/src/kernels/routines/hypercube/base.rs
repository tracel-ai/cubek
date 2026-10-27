use crate::kernels::{
    AttentionKernel, AttentionRoutine, AttentionTensors,
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeSecondLaunchFragmentAttentionRoutine,
    HypercubeSecondLaunchShuffleAttentionRoutine,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::hypercube::HypercubeAttentionMicrokernel;
use cubek_microkernel::attention::plane::CubePlaneMicrokernel;
use cubek_std::Hardware;
use serde::{Deserialize, Serialize};

/// The hypercube routines a pin names, each at its lead setting, at any query length: a box's
/// keys cut across cubes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum HypercubeAttentionKernel {
    /// [`HypercubeLastArrivalShuffleAttentionRoutine`]: each cube's planes shuffling their part
    /// of its run, over a float cache or a packed one decoded as it is read, the last cube of a
    /// box merging it.
    LastArrivalShuffle,
    /// [`HypercubeSecondLaunchShuffleAttentionRoutine`]: the same cubes, their slots merged by a
    /// second launch.
    SecondLaunchShuffle,
    /// [`HypercubeSecondLaunchFragmentAttentionRoutine`], over a float cache: each cube staging
    /// its run for plane fragments on a square matrix instruction, its slots merged by a second
    /// launch.
    SecondLaunchFragment,
}

/// How the slots of a box are merged into its queries of the output, by
/// [`plane_split_keys_merge_microkernel()`](fn@cubek_microkernel::split_keys_merge::plane::plane_split_keys_merge_microkernel)
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

impl HypercubeAttentionKernel {
    /// Every hypercube routine a pin names, in declaration order.
    pub fn all() -> Vec<Self> {
        vec![
            Self::LastArrivalShuffle,
            Self::SecondLaunchShuffle,
            Self::SecondLaunchFragment,
        ]
    }

    /// What a rule that named this routine runs where it declines on geometry the rules cannot
    /// see: the shuffle merged by a second launch after the one merged on the last arrival, which
    /// declines where one cube's stores do not reach another within a launch, and the unit routine
    /// after any other.
    pub fn instead_by_rule(&self) -> AttentionKernel {
        match self {
            Self::LastArrivalShuffle => AttentionKernel::Hypercube(Self::SecondLaunchShuffle),
            Self::SecondLaunchShuffle | Self::SecondLaunchFragment => AttentionKernel::Unit,
        }
    }

    /// What a report calls this routine, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::LastArrivalShuffle => "hypercube-last-arrival-shuffle-attention",
            Self::SecondLaunchShuffle => "hypercube-second-launch-shuffle-attention",
            Self::SecondLaunchFragment => "hypercube-second-launch-fragment-attention",
        }
    }

    /// This routine's lead on `tensors` ([`AttentionRoutine::launch_lead_on`]).
    ///
    /// # Errors
    ///
    /// Why it declines this problem on this device.
    pub fn launch_lead_on(
        &self,
        hw: &Hardware,
        tensors: AttentionTensors,
    ) -> Result<TensorHandle, Refusal> {
        match self {
            Self::LastArrivalShuffle => {
                HypercubeLastArrivalShuffleAttentionRoutine::new().launch_lead_on(hw, tensors)
            }
            Self::SecondLaunchShuffle => {
                HypercubeSecondLaunchShuffleAttentionRoutine::new().launch_lead_on(hw, tensors)
            }
            Self::SecondLaunchFragment => {
                HypercubeSecondLaunchFragmentAttentionRoutine.launch_lead_on(hw, tensors)
            }
        }
    }
}

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

    /// The queries one plane merging this way holds: a box's of `microkernel` for the last
    /// arrival, one for a second launch.
    pub fn seq_q<M: CubePlaneMicrokernel>(
        &self,
        microkernel: &HypercubeAttentionMicrokernel<M>,
    ) -> usize {
        match self {
            Self::LastArrival => microkernel.cube.seq_q(),
            Self::SecondLaunch => 1,
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
