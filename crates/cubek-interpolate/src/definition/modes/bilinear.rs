use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bilinear {}

#[cube]
impl Interpolate for Bilinear {
    const HALO: usize = 2;

    fn compute_weights<F: Float, N: Size>(frac: F) -> Array<Vector<F, N>> {
        let inverse_frac = F::one() - frac;

        let mut weights = Array::<Vector<F, N>>::new(Self::HALO);

        weights[0] = Vector::cast_from(inverse_frac);
        weights[1] = Vector::cast_from(frac);

        weights
    }
}
