use cubecl::prelude::*;

#[cube]
pub trait Interpolate {
    fn halo() -> comptime_type!(usize);

    fn compute_weights<F: Float, N: Size>(
        frac_x: F,
        frac_y: F,
    ) -> (Array<Vector<F, N>>, Array<Vector<F, N>>);
}
