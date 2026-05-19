use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bilinear {}

#[cube]
impl Interpolate for Bilinear {
    const HALO: usize = 2;

    fn compute_weights<F: Float, N: Size>(
        frac_x: F,
        frac_y: F,
    ) -> (Array<Vector<F, N>>, Array<Vector<F, N>>) {
        let inverse_frac_x = F::one() - frac_x;
        let inverse_frac_y = F::one() - frac_y;

        let mut weights_x = Array::<Vector<F, N>>::new(Self::HALO);
        let mut weights_y = Array::<Vector<F, N>>::new(Self::HALO);

        weights_x[0] = Vector::cast_from(inverse_frac_x);
        weights_x[1] = Vector::cast_from(frac_x);
        weights_y[0] = Vector::cast_from(inverse_frac_y);
        weights_y[1] = Vector::cast_from(frac_y);

        (weights_x, weights_y)
    }
}
