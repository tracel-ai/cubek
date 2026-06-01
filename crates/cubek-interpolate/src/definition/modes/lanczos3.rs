use crate::{
    components::readers::ReaderType,
    definition::{Interpolate, InterpolatePrecision, compute_value_default},
};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Lanczos3 {}

#[cube]
impl Interpolate for Lanczos3 {
    const HALO: usize = 6;

    const REQUIRES_BOUND_CHECK: bool = true;

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

    fn compute_value<P: InterpolatePrecision, N: Size>(
        input: &Tensor<Vector<P::EI, N>>,
        input_height: usize,
        input_width: usize,
        base_row: isize,
        base_col: isize,
        frac_row: P::EA,
        frac_col: P::EA,
        reader: ReaderType<P::EA, N>,
    ) -> Vector<P::EI, N> {
        compute_value_default::<Self, P, N>(
            input,
            input_height,
            input_width,
            base_row,
            base_col,
            frac_row,
            frac_col,
            reader,
        )
    }
}
