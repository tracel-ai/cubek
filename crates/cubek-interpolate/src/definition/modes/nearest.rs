use crate::{
    components::readers::ReaderType,
    definition::{Interpolate, InterpolatePrecision},
};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Nearest {}

#[cube]
impl Interpolate for Nearest {
    const HALO: usize = 1;

    const REQUIRES_BOUND_CHECK: bool = false;

    fn compute_weight<EA: Float>(_x: EA) -> EA {
        EA::new(1.0)
    }

    fn compute_value<P: InterpolatePrecision, N: Size>(
        input: &Tensor<Vector<P::EI, N>>,
        input_height: usize,
        input_width: usize,
        base_row: isize,
        base_col: isize,
        _frac_row: P::EA,
        _frac_col: P::EA,
        reader: ReaderType<P::EI, N>,
    ) -> Vector<P::EI, N> {
        let clamped_row = base_row.max(0).min(input_height as isize - 1) as usize;
        let clamped_col = base_col.max(0).min(input_width as isize - 1) as usize;

        Vector::cast_from(reader.read_weighted::<P::EA>(
            input,
            clamped_row,
            clamped_col,
            Vector::cast_from(P::EA::new(1.0)),
        ))
    }
}
