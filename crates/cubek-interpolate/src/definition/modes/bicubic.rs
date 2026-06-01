use crate::{
    components::readers::ReaderType,
    definition::{Interpolate, InterpolatePrecision, compute_value_default},
};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bicubic {}

#[cube]
impl Interpolate for Bicubic {
    const HALO: usize = 4;

    const REQUIRES_BOUND_CHECK: bool = false;

    fn compute_weight<EA: Float>(x: EA) -> EA {
        let a = EA::new(-0.75);
        let abs_x = x.abs();

        let x2 = abs_x * abs_x;
        let x3 = x2 * abs_x;

        // Convolution 1 (|x| <= 1.0)
        let w1 = (a + EA::new(2.0)) * x3 - (a + EA::new(3.0)) * x2 + EA::new(1.0);

        // Convolution 2 (1.0 < |x| <= 2.0)
        let w2 = a * x3 - EA::new(5.0) * a * x2 + EA::new(8.0) * a * abs_x - EA::new(4.0) * a;

        select(
            abs_x <= EA::new(1.0),
            w1,
            select(abs_x <= EA::new(2.0), w2, EA::new(0.0)),
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
