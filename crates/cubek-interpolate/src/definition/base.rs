use crate::definition::Transform;
use cubecl::prelude::CubeType;
use serde::{Deserialize, Serialize};

/// Algorithm used for upsampling.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType, Serialize, Deserialize)]
pub enum InterpolateMode {
    Nearest(NearestMode),
    Bilinear,
    Bicubic,
    Lanczos3,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType, Serialize, Deserialize)]
pub enum NearestMode {
    Exact,
    Floor,
}

/// Interpolation options.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InterpolateOptions {
    pub mode: InterpolateMode,
    pub align_corners: bool,
}

impl InterpolateOptions {
    pub fn new(mode: InterpolateMode) -> Self {
        Self {
            mode,
            align_corners: true,
        }
    }

    pub fn with_align_corners(mut self, align_corners: bool) -> Self {
        self.align_corners = align_corners;
        self
    }
}

/// Calculate the transform for the given input and output sizes and options.
pub fn get_transform(
    input_size: usize,
    output_size: usize,
    options: InterpolateOptions,
) -> Transform {
    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => match nearest_mode {
            NearestMode::Exact => Transform {
                scale_numerator: input_size,
                scale_denominator: output_size,
                offset_numerator: input_size as isize,
                offset_denominator: (2 * output_size) as isize,
            },
            NearestMode::Floor => Transform {
                scale_numerator: input_size,
                scale_denominator: output_size,
                offset_numerator: 0,
                offset_denominator: 1,
            },
        },
        _ if options.align_corners => Transform {
            scale_numerator: input_size.saturating_sub(1),
            scale_denominator: output_size.saturating_sub(1).max(1),
            offset_numerator: 0,
            offset_denominator: 1,
        },
        _ => Transform {
            scale_numerator: input_size,
            scale_denominator: output_size,
            offset_numerator: input_size as isize - output_size as isize,
            offset_denominator: (2 * output_size) as isize,
        },
    }
}
