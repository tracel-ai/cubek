use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Nearest {}

#[cube]
impl Interpolate for Nearest {
    const HALO: usize = 1;

    fn compute_weights<F: Float, N: Size>(_frac: F) -> Array<Vector<F, N>> {
        let mut weights = Array::<Vector<F, N>>::new(Self::HALO);
        weights[0] = Vector::cast_from(F::one());
        weights
    }
}
