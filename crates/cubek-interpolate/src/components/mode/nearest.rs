use super::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Nearest {}

const NEAREST_HALO: usize = 1;

#[cube]
impl Interpolate for Nearest {
    fn halo() -> comptime_type!(usize) {
        NEAREST_HALO
    }

    fn compute_weights<F: Float, N: Size>(_frac: F, weights: &mut Array<Vector<F, N>>) {
        weights[0] = Vector::cast_from(1);
    }
}
