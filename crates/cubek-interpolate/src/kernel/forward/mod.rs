mod bicubic;
mod bilinear;
mod lanczos3;
mod nearest;

use cubecl::prelude::*;

pub(crate) use bicubic::interpolate_bicubic_launch;
pub(crate) use bilinear::interpolate_bilinear_launch;
pub(crate) use lanczos3::interpolate_lanczos3_launch;
pub(crate) use nearest::interpolate_nearest_launch;

#[cube]
pub(crate) fn get_pixel_fraction(x: usize, ratio: f32, #[comptime] align_corners: bool) -> f32 {
    if align_corners {
        x as f32 * ratio
    } else {
        (x as f32 + 0.5) * ratio - 0.5
    }
}

#[cube]
pub(crate) fn get_ratio(
    input_size: usize,
    output_size: usize,
    #[comptime] align_corners: bool,
) -> f32 {
    if align_corners {
        let input_size = input_size.saturating_sub(1) as f32;
        let output_size = clamp_min(output_size - 1, 1) as f32;
        input_size / output_size
    } else {
        input_size as f32 / output_size as f32
    }
}
