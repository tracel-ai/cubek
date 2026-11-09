use super::{CubeStagedConvRoutine, UnitRegistersConvRoutine};
use crate::kernels::tiled::{ConvRoutine, ConvSettings, ConvTensors, launch_tuned};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;
use cubek_std::routine::Routine;

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

/// What a launch is declined with where it pins a cube staged setting on the unit registers
/// routine, whose settings are its own.
pub const CONV_SETTINGS_NOT_UNIT: Refusal = Refusal::Declined {
    why: "the unit registers routine takes none of the cube staged settings",
};

/// The routines a caller names, each at a setting or at the one its autotune finds fastest.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvRoutineKind {
    /// [`CubeStagedConvRoutine`], and where no setting is named on a device offering no plane
    /// instruction, [`UnitRegistersConvRoutine`] in its place.
    CubeStaged,
    /// [`UnitRegistersConvRoutine`] at the first of its settings that serves the convolution.
    UnitRegisters,
}

impl core::fmt::Display for ConvRoutineKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeStaged => write!(f, "cube_staged"),
            Self::UnitRegisters => write!(f, "unit_registers"),
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
                None if CubeStagedConvRoutine.settings(hw).is_empty() => {
                    UnitRegistersConvRoutine.launch_first_served(hw, tensors)
                }
                None => launch_tuned(hw, tensors),
            },
            Self::UnitRegisters => match settings {
                Some(_) => Err(CONV_SETTINGS_NOT_UNIT),
                None => UnitRegistersConvRoutine.launch_first_served(hw, tensors),
            },
        }
    }
}
