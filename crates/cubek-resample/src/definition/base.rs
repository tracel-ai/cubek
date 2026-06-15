use crate::definition::{Kernel, Placement, Semiring};
use cubecl::prelude::*;

/// Resampling operation.
#[derive(Debug, Clone, PartialEq, Eq, Hash, CubeType)]
pub struct Resample {
    pub resample_axes: Sequence<ResampleAxis>,
    pub semiring: Semiring,
}

impl Resample {
    pub fn new(semiring: Semiring) -> Self {
        Self {
            resample_axes: Sequence::new(),
            semiring,
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
