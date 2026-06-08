use crate::components::{Layout, LayoutExpand, NdLayout};
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

#[derive(CubeType, Clone)]
pub struct ReduceAxisPattern {
    pub reduce_size: u32,
}

#[cube]
impl AccessPattern for ReduceAxisPattern {
    fn footprint_size(access_pattern: &Self) -> u32 {
        access_pattern.reduce_size
    }

    fn read_values<C: Numeric>(
        input: &Tensor<C>,
        in_layout: &NdLayout,
        out_coord: &CoordsDyn,
        _tap_idx: u32,
        _access_pattern: &Self,
    ) -> TapResult<C> {
        let input_idx = in_layout.to_source_pos(&out_coord);

        let value = input[input_idx];

        let weight = C::from_int(1);

        TapResult::<C> { value, weight }
    }
}
