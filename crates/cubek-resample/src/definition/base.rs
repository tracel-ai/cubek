use crate::definition::{Kernel, Placement, PlacementArg, Semiring};
use cubecl::prelude::*;

/// Resampling operation.
#[derive(CubeType, CubeLaunch)]
pub struct ResampleArgs {
    pub placement_args: Sequence<PlacementArg>,
}

impl ResampleArgs {
    pub fn new() -> Self {
        Self {
            placement_args: Sequence::new(),
        }
    }

    pub fn with_placement_arg(mut self, placement_arg: PlacementArg) -> Self {
        self.placement_args.push(placement_arg);
        self
    }

    pub fn to_launch<R: Runtime>(self) -> ResampleArgsLaunch<R> {
        let mut placement_args = SequenceArg::new();
        for placement_arg in self.placement_args.iter() {
            placement_args.push(placement_arg.to_launch::<R>());
        }
        ResampleArgsLaunch::new(placement_args)
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
