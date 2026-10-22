use crate::kernels::{
    AttentionKernel, AttentionRoutine, AttentionTensors,
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeSecondLaunchFragmentAttentionRoutine,
    HypercubeSecondLaunchShuffleAttentionRoutine,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
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
