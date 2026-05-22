use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Lanczos3 {}

#[cube]
impl Interpolate for Lanczos3 {
    const HALO: usize = 6;

    fn compute_weight<EA: Float>(x: EA) -> EA {
        let abs_x = x.abs();
        let pi_x = EA::cast_from(core::f32::consts::PI) * x;
        let denom = (pi_x * pi_x) / EA::new(3.0);
        let safe_denom = select(abs_x < EA::new(1e-7), EA::new(1.0), denom);

        select(
            abs_x < EA::new(1e-7),
            EA::new(1.0),
            select(
                abs_x < EA::new(3.0),
                (pi_x.sin() * (pi_x / EA::new(3.0)).sin()) / safe_denom,
                EA::new(0.0),
            ),
        )
    }
}
