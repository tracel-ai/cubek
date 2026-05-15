use cubecl::prelude::*;

#[cube]
pub trait Interpolate {
    fn halo() -> comptime_type!(usize);

    fn compute_weights<F: Float, N: Size>(frac: f32, weights: &mut Array<Vector<F, N>>);
}
