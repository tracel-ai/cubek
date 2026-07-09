use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bilinear {}

#[cube]
impl Interpolate for Bilinear {
    const HALO: usize = 2;

    const REQUIRES_BOUND_CHECK: bool = false;

    fn compute_weight<EA: Float>(x: EA) -> EA {
        let abs_x = x.abs();
        select(
            abs_x < EA::new(1.0_f32),
            EA::new(1.0_f32) - abs_x,
            EA::new(0.0_f32),
        )
    }
}
