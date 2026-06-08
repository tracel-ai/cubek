use crate::components::NdLayout;
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};

#[derive(CubeType)]
pub struct TapResult<C: Numeric> {
    pub value: C,
    pub weight: C,
}

#[cube]
pub trait AccessPattern: CubeType + Clone {
    fn footprint_size(access_pattern: &Self) -> u32;

    fn read_values<C: Numeric>(
        input: &Tensor<C>,
        in_layout: &NdLayout,
        out_coord: &CoordsDyn,
        tap_idx: u32,
        access_pattern: &Self,
    ) -> TapResult<C>;
}
