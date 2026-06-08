use crate::components::{AccessPattern, NdLayout, TapResult};
use cubecl::{prelude::*, std::tensor::layout::CoordsDyn};
#[cube]
pub trait MemoryReader<P: AccessPattern>: CubeType {
    fn init(out_coord: CoordsDyn, args: P) -> Self;

    fn read_at<C: Numeric>(
        &self,
        input: &Tensor<C>,
        in_layout: &NdLayout,
        tap_idx: u32,
    ) -> TapResult<C>;

    fn num_taps(&self) -> u32;
}

#[derive(CubeType)]
pub struct GlobalReader<P: AccessPattern> {
    tap_idx: u32,
    total_taps: u32,
    out_coord: CoordsDyn,
    access_pattern: P,
}

#[cube]
impl<P: AccessPattern + Clone> MemoryReader<P> for GlobalReader<P> {
    fn init(out_coord: CoordsDyn, access_pattern: P) -> Self {
        let total_taps = P::footprint_size(&access_pattern);
        GlobalReader::<P> {
            tap_idx: 0,
            total_taps,
            out_coord,
            access_pattern,
        }
    }

    fn read_at<C: Numeric>(
        &self,
        input: &Tensor<C>,
        in_layout: &NdLayout,
        tap_idx: u32,
    ) -> TapResult<C> {
        P::read_values(
            input,
            in_layout,
            &self.out_coord,
            tap_idx,
            &self.access_pattern,
        )
    }

    fn num_taps(&self) -> u32 {
        self.total_taps
    }
}
