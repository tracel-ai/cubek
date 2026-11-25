use super::{CubeStagedConvRoutine, PlaneWindowConvRoutine};
use crate::kernels::tiled::{ConvRoutine, ConvTensors, CubeStagedConvSettings};
use cubecl::tile::launch::Decline;
use cubek_std::Hardware;

/// The routines a caller names, each at a setting or at the first of its own that serves. None
/// measures: a caller that does races every
/// [`CubeStagedConvCandidate`](crate::kernels::tiled::CubeStagedConvCandidate) the device lists.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ConvRoutineKind {
    /// `CubeStagedConvRoutine` at the setting named, or at the first of its settings that serves
    /// the convolution where none is.
    CubeStaged(Option<CubeStagedConvSettings>),
    /// `PlaneWindowConvRoutine` at the first of its settings that serves the convolution.
    PlaneWindow,
}

impl core::fmt::Display for ConvRoutineKind {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::CubeStaged(None) => write!(f, "cube_staged"),
            Self::CubeStaged(Some(settings)) => write!(f, "cube_staged_{settings}"),
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
    pub fn launch_on(&self, hw: &Hardware, tensors: ConvTensors) -> Result<(), Decline> {
        if tensors.is_empty() {
            return Ok(());
        }
        match self {
            Self::CubeStaged(Some(pinned)) => CubeStagedConvRoutine.launch_on(hw, tensors, pinned),
            Self::CubeStaged(None) => CubeStagedConvRoutine.launch_first_served(hw, tensors),
            Self::PlaneWindow => PlaneWindowConvRoutine.launch_first_served(hw, tensors),
        }
    }
}
