use cubecl::prelude::CubeType;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, CubeType)]
pub struct Transform {
    pub scale_numerator: usize,
    pub scale_denominator: usize,
    pub offset_numerator: isize,
    pub offset_denominator: isize,
}
