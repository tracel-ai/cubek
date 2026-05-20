use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bilinear {}

#[cube]
impl Interpolate for Bilinear {
    const HALO: usize = 2;

    fn compute_weight(x: f32) -> f32 {
        let abs_x = f32::abs(x);
        select(abs_x < 1.0, 1.0 - abs_x, 0.0)
    }
}
