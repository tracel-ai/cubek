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
pub(crate) fn get_pixel_fraction(
    input_width: usize,
    output_width: usize,
    x: usize,
    #[comptime] align_corners: bool,
) -> f32 {
    if align_corners {
        let input_width = input_width.saturating_sub(1) as f32;
        let output_width = clamp_min(output_width - 1, 1) as f32;
        (x as f32 * input_width) / output_width
    } else {
        let in_size = input_width as f32;
        let out_size = output_width as f32;
        (x as f32 + 0.5) * (in_size / out_size) - 0.5
    }
}
