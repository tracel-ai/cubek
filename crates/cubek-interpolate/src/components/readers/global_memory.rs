use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
#[expand(derive(Clone, Copy))]
pub struct GlobalMemoryReader {
    base_offset: usize,
    vector_size: usize,
    input_height: usize,
    input_width: usize,
}

#[cube]
impl GlobalMemoryReader {
    pub fn new<EI: Float, N: Size>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        input_height: usize,
        input_width: usize,
        #[comptime] vector_size: usize,
    ) -> Self {
        let base_offset = batch * input.stride(0) + channel_group * input.stride(3) * vector_size;

        GlobalMemoryReader {
            base_offset,
            vector_size,
            input_height,
            input_width,
        }
    }

    pub fn read_weighted<EI: Float, EA: Float, N: Size>(
        &self,
        input: &Tensor<Vector<EI, N>>,
        row: usize,
        col: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let input_idx = (self.base_offset
            + row.min(self.input_height - 1) * input.stride(1)
            + col.min(self.input_width - 1) * input.stride(2))
            / self.vector_size;

        Vector::cast_from(input[input_idx]) * weight
    }
}
