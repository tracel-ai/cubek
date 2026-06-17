use crate::definition::{Kernel, Placement, Semiring};
use cubecl::prelude::*;

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
