use crate::definition::{Placement, ResampleAxis};
use cubecl::prelude::*;

/// Footprint of a resample kernel.
#[derive(CubeType)]
pub struct Footprint<F: Float> {
    pub tap_scale: F,
    pub tap_offset: F,
}

#[cube]
impl<F: Float> Footprint<F> {
    pub fn new(#[comptime] axis: &ResampleAxis) -> Footprint<F> {
        let (tap_scale, tap_offset) = match axis.placement {
            Placement::Continuous { scale, offset } => (scale, offset),
            Placement::Windowed { step, pad } => (1.0 / (step as f32), -(pad as f32)),
        };

        Footprint::<F> {
            tap_scale: F::new(tap_scale),
            tap_offset: F::new(tap_offset),
        }
    }

    pub fn start_tap_and_frac(&self, radius: usize, pos: usize) -> (isize, F) {
        let center = F::cast_from(pos) * self.tap_scale + self.tap_offset;
        let center_floored = center.floor();

        let frac = center - center_floored;

        let start_tap = isize::cast_from(center_floored) - radius as isize + 1;

        (start_tap, frac)
    }
}
