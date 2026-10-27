use crate::kernels::{
    AttentionKernel, AttentionRoutine, AttentionTensors, CubeShuffleAttentionRoutine,
    CubeStagedAttentionRoutine,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use serde::{Deserialize, Serialize};

/// The cube routines a pin names, each at its lead setting, at any query length, over a float
/// cache.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CubeAttentionKernel {
    /// [`CubeStagedAttentionRoutine`]: a cube's planes on a square matrix instruction, against the
    /// keys the cube stages.
    Fragment,
    /// [`CubeShuffleAttentionRoutine`]: a cube's planes reading their keys where they lie, its
    /// units scoring a key in teams and summing across the plane.
    Shuffle,
}

impl CubeAttentionKernel {
    /// Every cube routine a pin names, in declaration order.
    pub fn all() -> Vec<Self> {
        vec![Self::Fragment, Self::Shuffle]
    }

    /// What a rule that named this routine runs where it declines on geometry the rules cannot
    /// see: the plane shuffles after the plane fragments, which decline an element with no square
    /// instruction, and the unit routine after the shuffles.
    pub fn instead_by_rule(&self) -> AttentionKernel {
        match self {
            Self::Fragment => AttentionKernel::Cube(Self::Shuffle),
            Self::Shuffle => AttentionKernel::Unit,
        }
    }

    /// What a report calls this routine, in kebab-case.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Fragment => "cube-fragment-attention",
            Self::Shuffle => "cube-shuffle-attention",
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
            Self::Fragment => CubeStagedAttentionRoutine.launch_lead_on(hw, tensors),
            Self::Shuffle => CubeShuffleAttentionRoutine.launch_lead_on(hw, tensors),
        }
    }
}
