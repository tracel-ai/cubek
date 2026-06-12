use crate::definition::{Kernel, Semiring};
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

/// Coordinate map: output index to source coordinate.
#[derive(Debug, Clone, Copy, PartialEq, CubeType)]
pub enum Placement {
    /// Continuous affine slide: `start = scale * pos + offset`.
    Continuous { scale: f32, offset: f32 },
    /// Windowed: `start = step * pos − pad`.
    Windowed { step: usize, pad: usize },
}

impl Eq for Placement {}

// Hash implementation to fix f32 `#[derive(Hash)]` error.
impl core::hash::Hash for Placement {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        core::mem::discriminant(self).hash(state);
        match self {
            Placement::Continuous { scale, offset } => {
                scale.to_bits().hash(state);
                offset.to_bits().hash(state);
            }
            Placement::Windowed { step, pad } => {
                step.hash(state);
                pad.hash(state);
            }
        }
    }
}
