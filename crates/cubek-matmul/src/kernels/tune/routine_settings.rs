//! One routine of the matmul at one of its settings, named so that something other than autotune
//! can launch it again: what an autotune's winner is identified by.

use crate::kernels::plan::MatmulCall;
use crate::kernels::{
    CubeShuffleMatmulRoutine, CubeShuffleMatmulRoutineSettings, CubeStagedAsyncCopyMatmulRoutine,
    CubeStagedAsyncCopyMatmulRoutineSettings, CubeStagedMatmulRoutine,
    CubeStagedMatmulRoutineSettings, CubeStagedTmaFillerMatmulRoutine,
    CubeStagedTmaFillerMatmulRoutineSettings, CubeStagedTmaMatmulRoutine,
    CubeStagedTmaMatmulRoutineSettings, CubeStagedTmaStoredMatmulRoutine,
    CubeStagedTmaStoredMatmulRoutineSettings, CubeStagedWgmmaMatmulRoutine,
    CubeStagedWgmmaMatmulRoutineSettings, HypercubeLastArrivalMatmulRoutine,
    HypercubeLastArrivalMatmulRoutineSettings, MatmulRoutine, PlaneShuffleMatmulRoutine,
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
    CubeStagedAsyncCopy(CubeStagedAsyncCopyMatmulRoutineSettings),
    CubeStagedTma(CubeStagedTmaMatmulRoutineSettings),
    CubeStagedTmaFiller(CubeStagedTmaFillerMatmulRoutineSettings),
    CubeStagedTmaStored(CubeStagedTmaStoredMatmulRoutineSettings),
    CubeStagedWgmma(CubeStagedWgmmaMatmulRoutineSettings),
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
            Self::CubeStagedAsyncCopy(settings) => {
                launch(CubeStagedAsyncCopyMatmulRoutine, call, &settings)
            }
            Self::CubeStagedTma(settings) => launch(CubeStagedTmaMatmulRoutine, call, &settings),
            Self::CubeStagedTmaFiller(settings) => {
                launch(CubeStagedTmaFillerMatmulRoutine, call, &settings)
            }
            Self::CubeStagedTmaStored(settings) => {
                launch(CubeStagedTmaStoredMatmulRoutine, call, &settings)
            }
            Self::CubeStagedWgmma(settings) => {
                launch(CubeStagedWgmmaMatmulRoutine, call, &settings)
            }
            Self::HypercubeLastArrival(settings) => {
                launch(HypercubeLastArrivalMatmulRoutine, call, &settings)
            }
        }
    }

    /// The routine at these settings, prepared on `call`'s inputs for a sink and handed to
    /// `visitor` ([`MatmulRoutine::sink_launch_on`]): what a caller storing the product through a
    /// sink of its own launches, once an autotune named the routine
    /// ([`MatmulAutotune::fastest_routine`](crate::kernels::tune::MatmulAutotune::fastest_routine)).
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
            Self::CubeStagedAsyncCopy(settings) => {
                sink_launch(CubeStagedAsyncCopyMatmulRoutine, call, &settings, visitor)
            }
            Self::CubeStagedTma(settings) => {
                sink_launch(CubeStagedTmaMatmulRoutine, call, &settings, visitor)
            }
            Self::CubeStagedTmaFiller(settings) => {
                sink_launch(CubeStagedTmaFillerMatmulRoutine, call, &settings, visitor)
            }
            Self::CubeStagedTmaStored(settings) => {
                sink_launch(CubeStagedTmaStoredMatmulRoutine, call, &settings, visitor)
            }
            Self::CubeStagedWgmma(settings) => {
                sink_launch(CubeStagedWgmmaMatmulRoutine, call, &settings, visitor)
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
            Self::CubeStagedAsyncCopy(settings) => settings.fmt(f),
            Self::CubeStagedTma(settings) => settings.fmt(f),
            Self::CubeStagedTmaFiller(settings) => settings.fmt(f),
            Self::CubeStagedTmaStored(settings) => settings.fmt(f),
            Self::CubeStagedWgmma(settings) => settings.fmt(f),
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

impl From<CubeStagedAsyncCopyMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: CubeStagedAsyncCopyMatmulRoutineSettings) -> Self {
        Self::CubeStagedAsyncCopy(settings)
    }
}

impl From<CubeStagedTmaMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: CubeStagedTmaMatmulRoutineSettings) -> Self {
        Self::CubeStagedTma(settings)
    }
}

impl From<CubeStagedTmaFillerMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: CubeStagedTmaFillerMatmulRoutineSettings) -> Self {
        Self::CubeStagedTmaFiller(settings)
    }
}

impl From<CubeStagedTmaStoredMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: CubeStagedTmaStoredMatmulRoutineSettings) -> Self {
        Self::CubeStagedTmaStored(settings)
    }
}

impl From<CubeStagedWgmmaMatmulRoutineSettings> for MatmulRoutineSettings {
    fn from(settings: CubeStagedWgmmaMatmulRoutineSettings) -> Self {
        Self::CubeStagedWgmma(settings)
    }
}
