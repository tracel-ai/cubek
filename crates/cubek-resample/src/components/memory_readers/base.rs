use crate::components::{AccessPattern, NdLayout, TapResult, ReduceAxisPattern};
use crate::definition::{AccessPatternKind, MemoryReaderKind};
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

#[cube]
pub fn num_taps(
    out_coord: CoordsDyn,
    #[comptime] access_pattern_kind: AccessPatternKind,
    #[comptime] reader_kind: MemoryReaderKind,
) -> u32 {
    match access_pattern_kind {
        AccessPatternKind::ReduceAxisPattern(args) => {
            let access_pattern = ReduceAxisPattern {
                reduce_size: args.reduce_size,
            };
            match reader_kind {
                MemoryReaderKind::Global => {
                    let reader = GlobalReader::<ReduceAxisPattern>::init(out_coord, access_pattern);
                    reader.num_taps()
                }
            }
        }
    }
}

#[cube]
pub fn read_at<C: Numeric>(
    input: &Tensor<C>,
    in_layout: &NdLayout,
    out_coord: CoordsDyn,
    tap_idx: u32,
    #[comptime] access_pattern_kind: AccessPatternKind,
    #[comptime] reader_kind: MemoryReaderKind,
) -> TapResult<C> {
    match access_pattern_kind {
        AccessPatternKind::ReduceAxisPattern(args) => {
            let access_pattern = ReduceAxisPattern {
                reduce_size: args.reduce_size,
            };
            match reader_kind {
                MemoryReaderKind::Global => {
                    let reader = GlobalReader::<ReduceAxisPattern>::init(out_coord, access_pattern);
                    reader.read_at(input, in_layout, tap_idx)
                }
            }
        }
    }
}
