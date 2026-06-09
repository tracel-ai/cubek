use crate::definition::{Placement, Resample};

use cubecl::prelude::*;

/// Footprint of a resample kernel.
#[derive(CubeType)]
pub struct Footprint<C: Float> {
    /// The first tap index.
    pub start_tap: isize,
    /// The fractional part of the center.
    pub frac: C,
}

#[cube]
pub fn get_footprint<C: Float>(
    #[comptime] config: &Resample,
    radius: usize,
    pos: usize,
) -> Footprint<C> {
    let (tap_scale, tap_offset) = match config.placement {
        Placement::Continuous { scale, offset } => (scale, offset),
        Placement::Windowed { step, pad } => (1.0 / (step as f32), -(pad as f32)),
    };

    let center = C::cast_from(pos) * C::new(tap_scale) + C::new(tap_offset);
    let center_floored = center.floor();
    let frac = center - center_floored;

    let start_tap = isize::cast_from(center_floored) - radius as isize + 1;

    Footprint::<C> { start_tap, frac }
}
