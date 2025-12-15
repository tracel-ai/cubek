use cubecl::prelude::*;
use cubecl::std::FastDivmod;

use crate::components::ConvolutionOperation;

#[derive(CubeType, CubeLaunch, Clone)]
pub struct RuntimeArgs {
    pub shape_k: u32,
    pub channels: u32,
    pub padded_channels: FastDivmod,
    #[cube(comptime)]
    pub operation: ConvolutionOperation,
}
