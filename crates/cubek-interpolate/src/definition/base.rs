use crate::{
    components::readers::ReaderType,
    definition::{Bicubic, Bilinear, InterpolatePrecision, Lanczos3, Nearest},
    routines::InterpolateBlueprint,
};
use cubecl::prelude::*;

// Base trait for interpolation algorithms.
#[cube]
pub trait Interpolate {
    const HALO: usize;

    const REQUIRES_BOUND_CHECK: bool;

    fn compute_weight<EA: Float>(x: EA) -> EA;

    fn compute_value<P: InterpolatePrecision, N: Size>(
        input: &Tensor<Vector<P::EI, N>>,
        input_height: usize,
        input_width: usize,
        base_row: isize,
        base_col: isize,
        frac_row: P::EA,
        frac_col: P::EA,
        reader: ReaderType<P::EI, N>,
    ) -> Vector<P::EI, N>;
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

#[cube]
#[allow(clippy::match_like_matches_macro)]
pub fn is_flattened(#[comptime] options: InterpolateOptions) -> bool {
    match options.mode {
        InterpolateMode::Nearest(_) => true,
        _ => false,
    }
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

// Maps InterpolateMode to the corresponding Interpolate implementation for compute_value.
#[cube]
pub fn compute_value<P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    input_height: usize,
    input_width: usize,
    base_row: isize,
    base_col: isize,
    frac_row: P::EA,
    frac_col: P::EA,
    reader: ReaderType<P::EI, N>,
    #[comptime] blueprint: InterpolateBlueprint,
) -> Vector<P::EI, N> {
    match blueprint.options.mode {
        InterpolateMode::Nearest(_) => Nearest::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
        ),
        InterpolateMode::Bilinear => Bilinear::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
        ),
        InterpolateMode::Bicubic => Bicubic::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
        ),
        InterpolateMode::Lanczos3 => Lanczos3::compute_value::<P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
        ),
    }
}

#[cube]
pub fn compute_value_default<I: Interpolate, P: InterpolatePrecision, N: Size>(
    input: &Tensor<Vector<P::EI, N>>,
    input_height: usize,
    input_width: usize,
    base_row: isize,
    base_col: isize,
    frac_row: P::EA,
    frac_col: P::EA,
    reader: ReaderType<P::EI, N>,
) -> Vector<P::EI, N> {
    let halo = I::HALO;
    let radius_offset = ((halo - 1) / 2) as isize;

    let mut col_weights = Array::<Vector<P::EA, N>>::new(halo);
    let mut row_weights = Array::<Vector<P::EA, N>>::new(halo);

    #[unroll]
    for i in 0..halo {
        let offset = P::EA::cast_from(radius_offset - i as isize);

        col_weights[i] = Vector::cast_from(I::compute_weight::<P::EA>(frac_col + offset));
        row_weights[i] = Vector::cast_from(I::compute_weight::<P::EA>(frac_row + offset));
    }

    let mut final_value = Vector::zeroed();
    let mut total_weight = Vector::<P::EA, N>::zeroed();

    #[unroll]
    for i in 0..halo {
        let mut row_value = Vector::zeroed();
        let mut row_weight_sum = Vector::<P::EA, N>::zeroed();

        let row = base_row + i as isize - radius_offset;

        #[unroll]
        for j in 0..halo {
            let col = base_col + j as isize - radius_offset;

            let clamped_row = row.max(0).min(input_height as isize - 1) as usize;
            let clamped_col = col.max(0).min(input_width as isize - 1) as usize;

            let weight_col = col_weights[j];
            let read_val =
                reader.read_weighted::<P::EA>(input, clamped_row, clamped_col, weight_col);

            if I::REQUIRES_BOUND_CHECK {
                let is_in_bounds = col >= 0
                    && col < input_width as isize
                    && row >= 0
                    && row < input_height as isize;

                row_value += select(is_in_bounds, read_val, Vector::zeroed());
                row_weight_sum += select(is_in_bounds, weight_col, Vector::zeroed());
            } else {
                row_value += read_val;
            }
        }

        let weight_row = row_weights[i];

        final_value += row_value * weight_row;

        if I::REQUIRES_BOUND_CHECK {
            total_weight += row_weight_sum * weight_row;
        }
    }

    if I::REQUIRES_BOUND_CHECK {
        let epsilon = Vector::cast_from(P::EA::new(1e-7));
        Vector::cast_from(final_value / total_weight.max(epsilon))
    } else {
        Vector::cast_from(final_value)
    }
}
