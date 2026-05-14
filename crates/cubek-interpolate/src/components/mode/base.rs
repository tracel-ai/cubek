use cubecl::prelude::*;

#[cube]
pub trait Interpolate {
    /// tells number of iterations in interpolation
    fn halo() -> usize;

    fn compute_weights<F: Float, N: Size>(frac: F, weights: &mut Array<Vector<F, N>>);
}
