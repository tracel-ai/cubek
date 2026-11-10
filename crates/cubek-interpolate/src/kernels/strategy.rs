//! What a caller can ask interpolate for.

use crate::kernels::{
    CubeInterpolateRoutine, CubeInterpolateRoutineSettings, CubeStagedInterpolateRoutine,
    InterpolateTensors, launch_interpolate_routine,
};
use cubecl::tile::launch::Refusal;
use cubek_std::Hardware;

/// How an interpolate call picks its kernel: a routine at the settings the selector reads off the
/// problem and the device ([`select`](super::select)), or one kernel named outright.
///
/// The two routines are what an autotune races. On a GPU they differ in how much of a cube one
/// problem occupies and in whether the gathered input is staged, the choice that swings a run by
/// up to 4x either way and is therefore measured rather than modelled. A CPU stages nothing, so
/// both run the same kernel there.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum InterpolateStrategy {
    /// [`CubeInterpolateRoutine`], its cube wide so more loads are in flight. Runs everywhere: it
    /// stages nothing, so no device refuses it for want of shared memory. The default.
    #[default]
    Cube,
    /// [`CubeStagedInterpolateRoutine`], its cube at the planes a device schedules at once and
    /// the grid spent instead; [`CubeInterpolateRoutine`] at the same settings where one tap
    /// leaves nothing to stage, or where the device cannot hold the window.
    CubeStaged,
    /// Run exactly this kernel, for measuring it. One that cannot run the problem is refused,
    /// never replaced.
    Specific(InterpolateKernel),
}

/// The interpolate kernels, each a routine at stated settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum InterpolateKernel {
    /// [`CubeInterpolateRoutine`].
    Cube(CubeInterpolateRoutineSettings),
    /// [`CubeStagedInterpolateRoutine`].
    CubeStaged(CubeInterpolateRoutineSettings),
}

impl InterpolateKernel {
    /// `tensors` resampled by this kernel on `hw`.
    ///
    /// # Errors
    ///
    /// Why its routine cannot run the problem here.
    pub fn launch(&self, hw: &Hardware, tensors: &InterpolateTensors) -> Result<(), Refusal> {
        match self {
            Self::Cube(settings) => {
                launch_interpolate_routine(&CubeInterpolateRoutine, hw, tensors, settings)
            }
            Self::CubeStaged(settings) => {
                launch_interpolate_routine(&CubeStagedInterpolateRoutine, hw, tensors, settings)
            }
        }
    }
}
