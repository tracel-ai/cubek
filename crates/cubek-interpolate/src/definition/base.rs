use crate::definition::{Bicubic, Bilinear, Lanczos3, Nearest};
use cubecl::prelude::*;

// Base trait for interpolation algorithms.
#[cube]
pub trait Interpolate {
    const HALO: usize;

    fn compute_weights<F: Float, N: Size>(frac: F) -> Array<Vector<F, N>>;
}

/// Algorithm used for upsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum InterpolateMode {
    /// Nearest-neighbor interpolation.
    /// <https://en.wikipedia.org/wiki/Nearest-neighbor_interpolation>
    Nearest(NearestMode),

    /// Bilinear interpolation.
    /// <https://en.wikipedia.org/wiki/Bilinear_interpolation>
    Bilinear,

    /// Bicubic interpolation.
    /// <https://en.wikipedia.org/wiki/Bicubic_interpolation>
    Bicubic,

    /// Lanczos3 interpolation (6-tap sinc-based filter).
    /// <https://en.wikipedia.org/wiki/Lanczos_resampling>
    Lanczos3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub enum NearestMode {
    // Matches Scikit-Image and PIL nearest neighbours interpolation algorithms.
    Exact,
    // Matches buggy OpenCV’s INTER_NEAREST interpolation algorithm for backward compatibility.
    Floor,
}

// Helper functions to map InterpolateMode to the corresponding Interpolate implementation.
pub fn get_halo(mode: InterpolateMode) -> usize {
    match mode {
        InterpolateMode::Nearest(_) => <Nearest as Interpolate>::HALO,
        InterpolateMode::Bilinear => <Bilinear as Interpolate>::HALO,
        InterpolateMode::Bicubic => <Bicubic as Interpolate>::HALO,
        InterpolateMode::Lanczos3 => <Lanczos3 as Interpolate>::HALO,
    }
}

#[cube]
pub fn compute_weights<F: Float, N: Size>(
    frac: F,
    #[comptime] options: InterpolateOptions,
) -> Array<Vector<F, N>> {
    match options.mode {
        InterpolateMode::Nearest(_) => <Nearest as Interpolate>::compute_weights(frac),
        InterpolateMode::Bilinear => <Bilinear as Interpolate>::compute_weights(frac),
        InterpolateMode::Bicubic => <Bicubic as Interpolate>::compute_weights(frac),
        InterpolateMode::Lanczos3 => <Lanczos3 as Interpolate>::compute_weights(frac),
    }
}

/// Interpolation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateOptions {
    /// Algorithm used.
    pub mode: InterpolateMode,
    /// If `true`, the input and output tensors are aligned by their corner pixels.
    /// If `false`, half-pixel coordinate mapping is used instead.
    pub align_corners: bool,
}

impl InterpolateOptions {
    /// Create new interpolate options with the given mode.
    /// Defaults to `align_corners = true`.
    pub fn new(mode: InterpolateMode) -> Self {
        Self {
            mode,
            align_corners: true,
        }
    }

    /// Set align_corners.
    pub fn with_align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}
