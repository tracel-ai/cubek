use crate::definition::{Kernel, Placement, PlacementArgs, Semiring, WindowArgs};
use cubecl::prelude::*;

/// Resampling args.
#[derive(CubeType, CubeLaunch)]
pub struct ResampleArgs {
    pub resample_axes: Sequence<ResampleAxisArgs>,
}

impl ResampleArgs {
    pub fn new() -> Self {
        Self {
            resample_axes: Sequence::new(),
        }
    }

    pub fn with_resample_axis_args(mut self, resample_axis_args: ResampleAxisArgs) -> Self {
        self.resample_axes.push(resample_axis_args);
        self
    }

    pub fn to_launch<R: Runtime>(self) -> ResampleArgsLaunch<R> {
        let mut resample_axes_launch = SequenceArg::new();
        for resample_axes_args in self.resample_axes.iter() {
            resample_axes_launch.push(resample_axes_args.to_launch::<R>());
        }
        ResampleArgsLaunch::new(resample_axes_launch)
    }
}

/// Resampling axis args.
#[derive(CubeType, CubeLaunch)]
pub struct ResampleAxisArgs {
    pub window_args: WindowArgs,
    pub placement_args: PlacementArgs,
}

impl ResampleAxisArgs {
    pub fn new(window_args: WindowArgs, placement_args: PlacementArgs) -> Self {
        Self {
            window_args,
            placement_args,
        }
    }

    pub fn to_launch<R: Runtime>(&self) -> ResampleAxisArgsLaunch<R> {
        ResampleAxisArgsLaunch::new(
            self.window_args.to_launch::<R>(),
            self.placement_args.to_launch::<R>(),
        )
    }
}

/// Resampling operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub struct Resample {
    pub resample_axes: Sequence<ResampleAxis>,
    pub semiring: Semiring,
    pub boundary: BoundaryMode,
    pub normalization: NormalizationMode,
}

impl Resample {
    pub fn new(
        semiring: Semiring,
        boundary: BoundaryMode,
        normalization: NormalizationMode,
    ) -> Self {
        Self {
            resample_axes: Sequence::new(),
            semiring,
            boundary,
            normalization,
        }
    }

    /// Order matters, last axis added is innermost.
    pub fn with_axis(mut self, axis: ResampleAxis) -> Self {
        self.resample_axes.push(axis);
        self
    }
}

/// Resample axis operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub struct ResampleAxis {
    pub axis: usize,
    pub kernel: Kernel,
    pub placement: Placement,
}

impl ResampleAxis {
    pub fn new(axis: usize, kernel: Kernel, placement: Placement) -> Self {
        Self {
            axis,
            kernel,
            placement,
        }
    }
}

/// Boundary handling mode for out-of-bounds taps.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum BoundaryMode {
    /// Out-of-bounds taps contribute zero (skip the tap).
    Zero,
    /// Out-of-bounds coordinates are clamped to the nearest valid input coordinate.
    Clamp,
}

/// Normalization mode for tap weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum NormalizationMode {
    /// Preserve the kernel weights exactly.
    None,
    /// Divide by the accumulated valid weight.
    Renormalize,
}
