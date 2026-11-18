use super::{CubeStagedConvRoutine, PlaneWindowConvRoutine};
use crate::kernels::tiled::{ConvRoutine, ConvTensors, CubeStagedConvSettings, launch_tuned};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// The routines a caller names, each at a setting or at the one it finds on its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvRoutineKind {
    /// [`CubeStagedConvRoutine`] at the setting named, or at the first of its settings that serves
    /// the convolution where none is.
    CubeStaged(Option<CubeStagedConvSettings>),
    /// The setting the autotune finds fastest among those [`CubeStagedConvRoutine`] lists, and
    /// [`PlaneWindowConvRoutine`] where none of them serves the convolution.
    Tuned,
    /// [`PlaneWindowConvRoutine`] at the first of its settings that serves the convolution.
    PlaneWindow,
}

impl core::fmt::Display for ConvRoutineKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeStaged(None) => write!(f, "cube_staged"),
            Self::CubeStaged(Some(settings)) => write!(f, "cube_staged_{settings}"),
            Self::Tuned => write!(f, "tuned"),
            Self::PlaneWindow => write!(f, "plane_window"),
        }
    }
}

impl ConvRoutineKind {
    /// `tensors` launched on `hw` by this routine, at the setting it names or finds. Nothing is
    /// launched where the output is empty.
    ///
    /// # Errors
    ///
    /// Why the routine declines the convolution on `hw`.
    pub fn launch_on(&self, hw: &Hardware, tensors: ConvTensors) -> Result<(), Refusal> {
        if tensors.is_empty() {
            return Ok(());
        }
        match self {
            Self::CubeStaged(Some(pinned)) => CubeStagedConvRoutine.launch_on(hw, tensors, pinned),
            Self::CubeStaged(None) => CubeStagedConvRoutine.launch_first_served(hw, tensors),
            Self::Tuned => launch_tuned(hw, tensors),
            Self::PlaneWindow => PlaneWindowConvRoutine.launch_first_served(hw, tensors),
        }
    }
}
