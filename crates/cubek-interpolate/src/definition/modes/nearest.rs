use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Nearest {}

#[cube]
impl Interpolate for Nearest {
    const HALO: usize = 1;

    const REQUIRES_BOUND_CHECK: bool = false;

    fn compute_weight<EA: Float>(_x: EA) -> EA {
        EA::new(1.0_f32)
    }
}
