//! What a caller can ask random for.

use crate::kernels::{
    PlaneRandomRoutine, PlaneRandomRoutineSettings, RandomTensor, UnitRandomRoutine,
    UnitRandomRoutineSettings, launch_random_routine,
};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::random::{Distribution, SeedsLaunch};
use cubek_std::Hardware;

/// How a draw picks its kernel: the routine the selector reads off the device, or one kernel named
/// outright.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum RandomStrategy {
    /// [`PlaneRandomRoutine`] on a GPU, [`UnitRandomRoutine`] on a CPU, each at its default.
    #[default]
    Selected,
    /// Run exactly this kernel, for measuring it or holding it to the other.
    Specific(RandomKernel),
}

/// The random kernels, each a routine at stated settings.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RandomKernel {
    /// [`PlaneRandomRoutine`].
    Plane(PlaneRandomRoutineSettings),
    /// [`UnitRandomRoutine`].
    Unit(UnitRandomRoutineSettings),
}

impl RandomKernel {
    /// `tensor` filled by this kernel on `hw` with draws of `distribution` off `seeds`.
    ///
    /// # Errors
    ///
    /// Why its routine cannot run the problem here.
    pub fn launch<D: Distribution>(
        &self,
        hw: &Hardware,
        tensor: &RandomTensor,
        distribution: D,
        seeds: SeedsLaunch,
    ) -> Result<(), Refusal> {
        match self {
            Self::Plane(settings) => launch_random_routine(
                &PlaneRandomRoutine,
                hw,
                tensor,
                settings,
                distribution,
                seeds,
            ),
            Self::Unit(settings) => launch_random_routine(
                &UnitRandomRoutine,
                hw,
                tensor,
                settings,
                distribution,
                seeds,
            ),
        }
    }
}
