use crate::components::readers::{Reader, ReaderExpand};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct GlobalMemoryReader {
    base_offset: usize,
    vector_size: usize,
    input_width: usize,
    input_height: usize,
}

#[cube]
impl<EA: Float, N: Size> Reader<EA, N> for GlobalMemoryReader {
    fn prepare_read<EI: Float>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        input_width: usize,
        input_height: usize,
        _min_x: usize,
        _min_y: usize,
        #[comptime] vector_size: usize,
        #[comptime] _smem_width: usize,
        #[comptime] _smem_height: usize,
    ) -> Self {
        let base_offset = batch * input.stride(0) + channel_group * input.stride(3) * vector_size;

        GlobalMemoryReader {
            base_offset,
            vector_size,
            input_width,
            input_height,
        }
    }

    fn read_weighted<EI: Float>(
        &self,
        input: &Tensor<Vector<EI, N>>,
        y: usize,
        x: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let input_idx = (self.base_offset
            + y.max(0).min(self.input_height - 1) * input.stride(1)
            + x.max(0).min(self.input_width - 1) * input.stride(2))
            / self.vector_size;

        Vector::cast_from(input[input_idx]) * weight
    }
}
