use super::{
    CONV_SETTINGS_NOT_TUNED, CONV_SETTINGS_NOT_UNIT, CubeStagedConvRoutine,
    UnitRegistersConvRoutine,
};
use crate::kernels::tiled::{ConvRoutine, ConvSettings, ConvTensors, launch_tuned};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// The routines a caller names, each at a setting or at the one it finds on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvRoutineKind {
    /// [`CubeStagedConvRoutine`] at the setting named, or at the first of its settings that serves
    /// the convolution.
    CubeStaged,
    /// The setting the autotune finds fastest among those [`CubeStagedConvRoutine`] lists, and
    /// [`UnitRegistersConvRoutine`] where none of them serves the convolution.
    Tuned,
    /// [`UnitRegistersConvRoutine`] at the first of its settings that serves the convolution.
    UnitRegisters,
}

impl core::fmt::Display for ConvRoutineKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeStaged => write!(f, "cube_staged"),
            Self::Tuned => write!(f, "tuned"),
            Self::UnitRegisters => write!(f, "unit_registers"),
        }
    }
}

impl ConvRoutineKind {
    /// `tensors` launched on `hw` by this routine at `settings`, or at the setting it finds where
    /// none is named. Nothing is launched where the output is empty.
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
        if tensors.is_empty() {
            return Ok(());
        }
        match (self, settings) {
            (Self::CubeStaged, Some(settings)) => {
                CubeStagedConvRoutine.launch_on(hw, tensors, settings)
            }
            (Self::CubeStaged, None) => CubeStagedConvRoutine.launch_first_served(hw, tensors),
            (Self::Tuned, None) => launch_tuned(hw, tensors),
            (Self::Tuned, Some(_)) => Err(CONV_SETTINGS_NOT_TUNED),
            (Self::UnitRegisters, Some(_)) => Err(CONV_SETTINGS_NOT_UNIT),
            (Self::UnitRegisters, None) => {
                UnitRegistersConvRoutine.launch_first_served(hw, tensors)
            }
        }
    }
}
