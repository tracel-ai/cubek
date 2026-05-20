use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bicubic {}

#[cube]
impl Interpolate for Bicubic {
    const HALO: usize = 4;

    fn compute_weight(x: f32) -> f32 {
        let a = -0.75;
        let abs_x = f32::abs(x);

        let x2 = abs_x * abs_x;
        let x3 = x2 * abs_x;

        // Convolution 1 (|x| <= 1.0)
        let w1 = (a + 2.0) * x3 - (a + 3.0) * x2 + 1.0;

        // Convolution 2 (1.0 < |x| <= 2.0)
        let w2 = a * x3 - 5.0 * a * x2 + 8.0 * a * abs_x - 4.0 * a;

        select(abs_x <= 1.0, w1, select(abs_x <= 2.0, w2, 0.0))
    }
}
