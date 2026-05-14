use super::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Nearest {}

const NEAREST_HALO: usize = 1;

#[cube]
impl Interpolate for Nearest {
    fn halo() -> usize {
        NEAREST_HALO.into()
    }

    fn compute_weights<F: Float, N: Size>(_frac: F, weights: &mut Array<Vector<F, N>>) {
        weights[0] = Vector::<F, N>::cast_from(F::cast_from(1.0));
    }
}
