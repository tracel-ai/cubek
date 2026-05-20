use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Bilinear {}

#[cube]
impl Interpolate for Bilinear {
    const HALO: usize = 2;

    fn compute_weight<EA: Float>(x: EA) -> EA {
        let abs_x = x.abs();
        select(abs_x < EA::new(1.0), EA::new(1.0) - abs_x, EA::new(0.0))
    }
}
