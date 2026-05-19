use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Nearest {}

#[cube]
impl Interpolate for Nearest {
    const HALO: usize = 1;

    fn compute_weights<F: Float, N: Size>(
        _frac_x: F,
        _frac_y: F,
    ) -> (Array<Vector<F, N>>, Array<Vector<F, N>>) {
        let mut weights_x = Array::<Vector<F, N>>::new(Self::HALO);
        let mut weights_y = Array::<Vector<F, N>>::new(Self::HALO);
        weights_x[0] = Vector::cast_from(F::one());
        weights_y[0] = Vector::cast_from(F::one());
        (weights_x, weights_y)
    }
}
