use super::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bilinear {}

const BILINEAR_HALO: usize = 2;

#[cube]
impl Interpolate for Bilinear {
    fn halo() -> comptime_type!(usize) {
        BILINEAR_HALO
    }

    fn compute_weights<F: Float, N: Size>(frac: f32, weights: &mut Array<Vector<F, N>>) {
        let inverse_frac = 1.0 - frac;
        weights[0] = Vector::cast_from(inverse_frac);
        weights[1] = Vector::cast_from(frac);
    }
}
