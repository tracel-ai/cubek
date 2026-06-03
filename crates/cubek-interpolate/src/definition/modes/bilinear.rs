use crate::{
    components::readers::ReaderType,
    definition::{Interpolate, InterpolatePrecision, compute_value_default},
};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bilinear {}

#[cube]
impl Interpolate for Bilinear {
    const HALO: usize = 2;

    const REQUIRES_BOUND_CHECK: bool = false;

    fn compute_weight<EA: Float>(x: EA) -> EA {
        let abs_x = x.abs();
        select(abs_x < EA::new(1.0), EA::new(1.0) - abs_x, EA::new(0.0))
    }

    fn compute_value<P: InterpolatePrecision, N: Size>(
        input: &Tensor<Vector<P::EI, N>>,
        input_height: usize,
        input_width: usize,
        base_row: isize,
        base_col: isize,
        frac_row: P::EA,
        frac_col: P::EA,
        reader: ReaderType<P::EI, N>,
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
