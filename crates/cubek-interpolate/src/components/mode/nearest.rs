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

    fn compute_weights<F: Float, N: Size>(
        _frac_x: F,
        _frac_y: F,
    ) -> (Array<Vector<F, N>>, Array<Vector<F, N>>) {
        let mut weights_x = Array::<Vector<F, N>>::new(NEAREST_HALO);
        let mut weights_y = Array::<Vector<F, N>>::new(NEAREST_HALO);
        weights_x[0] = Vector::cast_from(F::one());
        weights_y[0] = Vector::cast_from(F::one());
        (weights_x, weights_y)
    }
}
