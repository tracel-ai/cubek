//! One routine of the matmul at one of its settings, named so that something other than autotune
//! can launch it again: what a table's winner is identified by.

use crate::kernels::plan::MatmulCall;
use crate::kernels::{
    CubeShuffleMatmulRoutine, CubeShuffleMatmulRoutineSettings, CubeStagedMatmulRoutine,
    CubeStagedMatmulRoutineSettings, HypercubeLastArrivalMatmulRoutine,
    HypercubeLastArrivalMatmulRoutineSettings, HypercubeRelayMatmulRoutine,
    HypercubeRelayMatmulRoutineSettings, MatmulRoutine, PlaneShuffleMatmulRoutine,
    PlaneShuffleMatmulRoutineSettings, UnitRegistersMatmulRoutine,
    UnitRegistersMatmulRoutineSettings, UnitWholeKMatmulRoutine, UnitWholeKMatmulRoutineSettings,
};
use cubecl::std::tensor::TensorHandle;
use cubecl::tile::launch::{Refusal, SinkLaunchVisitor};

/// A routine at one of its settings.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum MatmulRoutineSettings {
    UnitRegisters(UnitRegistersMatmulRoutineSettings),
    PlaneShuffle(PlaneShuffleMatmulRoutineSettings),
    UnitWholeK(UnitWholeKMatmulRoutineSettings),
    CubeShuffle(CubeShuffleMatmulRoutineSettings),
    CubeStaged(CubeStagedMatmulRoutineSettings),
    HypercubeRelay(HypercubeRelayMatmulRoutineSettings),
    HypercubeLastArrival(HypercubeLastArrivalMatmulRoutineSettings),
}

impl MatmulRoutineSettings {
    /// What a rule falls back to where its kernel declines, in order: the staged routine at its
    /// default, then the routine that asks nothing of the device, which serves any float product.
    pub const FALLBACK: [Self; 2] = [
        Self::CubeStaged(CubeStagedMatmulRoutineSettings::DEFAULT),
        Self::UnitRegisters(UnitRegistersMatmulRoutineSettings::DEFAULT),
    ];

    /// The routine at these settings, launched on `call`'s inputs as they are stored.
    ///
    /// # Errors
    ///
    /// Why the device cannot run the product at these settings.
    pub fn launch(self, call: &MatmulCall) -> Result<TensorHandle, Refusal> {
        match self {
            Self::UnitRegisters(settings) => launch(UnitRegistersMatmulRoutine, call, &settings),
            Self::PlaneShuffle(settings) => launch(PlaneShuffleMatmulRoutine, call, &settings),
            Self::UnitWholeK(settings) => launch(UnitWholeKMatmulRoutine, call, &settings),
            Self::CubeShuffle(settings) => launch(CubeShuffleMatmulRoutine, call, &settings),
            Self::CubeStaged(settings) => launch(CubeStagedMatmulRoutine, call, &settings),
            Self::HypercubeRelay(settings) => launch(HypercubeRelayMatmulRoutine, call, &settings),
            Self::HypercubeLastArrival(settings) => {
                launch(HypercubeLastArrivalMatmulRoutine, call, &settings)
            }
        }
    }

    /// The routine at these settings, prepared on `call`'s inputs for a sink and handed to
    /// `visitor` ([`MatmulRoutine::sink_launch_on`]): what a caller storing the product through a
    /// sink of its own launches, once a table named the routine
    /// ([`MatmulTable::fastest_routine`](crate::kernels::tune::MatmulTable::fastest_routine)).
    pub fn sink_launch<V: SinkLaunchVisitor>(self, call: &MatmulCall, visitor: V) -> V::Output {
        match self {
            Self::UnitRegisters(settings) => {
                sink_launch(UnitRegistersMatmulRoutine, call, &settings, visitor)
            }
            Self::PlaneShuffle(settings) => {
                sink_launch(PlaneShuffleMatmulRoutine, call, &settings, visitor)
            }
            Self::UnitWholeK(settings) => {
                sink_launch(UnitWholeKMatmulRoutine, call, &settings, visitor)
            }
            Self::CubeShuffle(settings) => {
                sink_launch(CubeShuffleMatmulRoutine, call, &settings, visitor)
            }
            Self::CubeStaged(settings) => {
                sink_launch(CubeStagedMatmulRoutine, call, &settings, visitor)
            }
            Self::HypercubeRelay(settings) => {
                sink_launch(HypercubeRelayMatmulRoutine, call, &settings, visitor)
            }
            Self::HypercubeLastArrival(settings) => {
                sink_launch(HypercubeLastArrivalMatmulRoutine, call, &settings, visitor)
            }
        }
    }
}

impl core::fmt::Display for MatmulRoutineSettings {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            Self::UnitRegisters(settings) => settings.fmt(f),
            Self::PlaneShuffle(settings) => settings.fmt(f),
            Self::UnitWholeK(settings) => settings.fmt(f),
            Self::CubeShuffle(settings) => settings.fmt(f),
            Self::CubeStaged(settings) => settings.fmt(f),
            Self::HypercubeRelay(settings) => settings.fmt(f),
            Self::HypercubeLastArrival(settings) => settings.fmt(f),
        }
    }
}

impl From<UnitRegistersMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: UnitRegistersMatmulRoutineSettings) -> Self {
        Self::UnitRegisters(settings)
    }
}

impl From<PlaneShuffleMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: PlaneShuffleMatmulRoutineSettings) -> Self {
        Self::PlaneShuffle(settings)
    }
}

impl From<UnitWholeKMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: UnitWholeKMatmulRoutineSettings) -> Self {
        Self::UnitWholeK(settings)
    }
}

impl From<CubeShuffleMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: CubeShuffleMatmulRoutineSettings) -> Self {
        Self::CubeShuffle(settings)
    }
}

impl From<CubeStagedMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: CubeStagedMatmulRoutineSettings) -> Self {
        Self::CubeStaged(settings)
    }
}

impl From<HypercubeRelayMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: HypercubeRelayMatmulRoutineSettings) -> Self {
        Self::HypercubeRelay(settings)
    }
}

impl From<HypercubeLastArrivalMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: HypercubeLastArrivalMatmulRoutineSettings) -> Self {
        Self::HypercubeLastArrival(settings)
    }
}

/// `routine` at `settings` on `call`'s inputs, written in the element they make.
fn launch<R, M, SM>(
    routine: R,
    call: &MatmulCall,
    settings: &R::Settings,
) -> Result<TensorHandle, Refusal>
where
    R: MatmulRoutine<M, SM>,
    M: cubek_microkernel::matmul::MatmulMicrokernel,
{
    routine.launch_on(
        &call.hw,
        &call.client,
        call.lhs.tensor.clone(),
        call.rhs.tensor.clone(),
        settings,
    )
}

/// `routine` at `settings` on `call`'s inputs, prepared for a sink and handed to `visitor`.
fn sink_launch<R, M, SM, V>(
    routine: R,
    call: &MatmulCall,
    settings: &R::Settings,
    visitor: V,
) -> V::Output
where
    R: MatmulRoutine<M, SM>,
    M: cubek_microkernel::matmul::MatmulMicrokernel,
    V: SinkLaunchVisitor,
{
    visitor.visit(routine.sink_launch_on(
        &call.hw,
        &call.client,
        call.lhs.tensor.clone(),
        call.rhs.tensor.clone(),
        settings,
    ))
}
