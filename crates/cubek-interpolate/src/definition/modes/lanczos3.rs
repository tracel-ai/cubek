use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Lanczos3 {}

#[cube]
impl Interpolate for Lanczos3 {
    const HALO: usize = 6;

    fn compute_weights<F: Float, N: Size>(frac: F) -> Array<Vector<F, N>> {
        let mut weights = Array::<Vector<F, N>>::new(Self::HALO);

        for i in 0..Self::HALO {
            let x = frac - F::cast_from(i as f32 - 2.0);
            weights[i] = Vector::new(F::cast_from(lanczos3_weight(f32::cast_from(x))));
        }

        weights
    }
}

#[cube]
fn lanczos3_weight(x: f32) -> f32 {
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
