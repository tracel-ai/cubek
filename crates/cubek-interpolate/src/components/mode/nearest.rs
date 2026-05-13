use super::Interpolate;
use cubecl::prelude::*;

pub struct Nearest {}

#[cube]
impl Interpolate for Nearest {
    fn halo() -> comtime_type!(usize) {
        // 1 as usize
        todo!()
    }

    fn compute_weights<F: Float, N: Size>(
        _x_fraction: Vector<F, N>,
        weights: &mut Array<Vector<F, N>>,
    ) {
        // weights[0] = Vector::cast_from(1.0);
        todo!()
    }
}
