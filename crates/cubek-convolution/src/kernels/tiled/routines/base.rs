use super::CubeStagedConvRoutine;
use crate::kernels::tiled::{ConvRoutine, ConvSettings, ConvTensors, launch_tuned};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// What a setting is declined with where the device offers no instruction of its kind for the
/// problem's elements.
pub const CONV_INSTRUCTION_UNOFFERED: Refusal = Refusal::Declined {
    why: "the device offers no instruction of this kind for the problem's elements",
};

/// What a setting is declined with where it asks for a warpgroup MMA, whose stages only the TMA
/// engine lands and which this convolution's fill never does.
pub const CONV_WGMMA_UNSERVED: Refusal = Refusal::Declined {
    why: "a warpgroup MMA reads its stages as the TMA engine lands them",
};

/// The routines a caller names, each at a setting or at the one its autotune finds fastest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvRoutineKind {
    /// [`CubeStagedConvRoutine`].
    CubeStaged,
}

impl core::fmt::Display for ConvRoutineKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeStaged => write!(f, "cube_staged"),
        }
    }
}

impl ConvRoutineKind {
    /// `tensors` launched on `hw` by this routine at `settings`, or at the setting its autotune
    /// finds fastest where none is named.
    ///
    /// # Errors
    ///
    /// Why the routine declines the convolution on `hw`.
    pub fn launch_on(
        &self,
        hw: &Hardware,
        tensors: ConvTensors,
        settings: Option<&ConvSettings>,
    ) -> Result<(), Refusal> {
        match self {
            Self::CubeStaged => match settings {
                Some(settings) => CubeStagedConvRoutine.launch_on(hw, tensors, settings),
                None => launch_tuned(hw, tensors),
            },
        }
    }
}
