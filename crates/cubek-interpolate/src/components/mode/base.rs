use cubecl::prelude::*;

#[cube]
pub trait Interpolate {
    /// tells number of iterations in interpolation
    fn halo() -> comptime_type!(usize);

    fn compute_weights<F: Float, N: Size>(frac: f32, weights: &mut Array<Vector<F, N>>);
}
