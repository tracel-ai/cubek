use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Lanczos3 {}

#[cube]
impl Interpolate for Lanczos3 {
    const HALO: usize = 6;

    fn compute_weight(x: f32) -> f32 {
        let abs_x = f32::abs(x);
        let pi_x = core::f32::consts::PI * x;
        let denom = (pi_x * pi_x) / 3.0;
        let safe_denom = select(abs_x < 1e-7, 1.0, denom);
        
        select(
            abs_x < 1e-7,
            1.0,
            select(
                abs_x < 3.0,
                (f32::sin(pi_x) * f32::sin(pi_x / 3.0)) / safe_denom,
                0.0,
            ),
        )
    }
}

