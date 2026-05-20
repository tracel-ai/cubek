use crate::definition::Interpolate;
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct Nearest {}

const NEAREST_WEIGHT: f32 = 1.0;

#[cube]
impl Interpolate for Nearest {
    const HALO: usize = 1;

    fn compute_weight(_x: f32) -> f32 {
        NEAREST_WEIGHT.into()
    }
}
