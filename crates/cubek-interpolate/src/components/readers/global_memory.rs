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
        input_height: usize,
        input_width: usize,
        #[comptime] vector_size: usize,
    ) -> Self {
        let base_offset = batch * input.stride(0);

        GlobalMemoryReader {
            base_offset,
            vector_size,
            input_height,
            input_width,
        }
    }

    pub fn read<EI: Float, N: Size>(
        &self,
        input: &Tensor<Vector<EI, N>>,
        row: usize,
        col: usize,
        vector_index: usize,
    ) -> Vector<EI, N> {
        let input_idx = (self.base_offset
            + row.min(self.input_height - 1) * input.stride(1)
            + col.min(self.input_width - 1) * input.stride(2))
            / self.vector_size
            + vector_index * input.stride(3);

        input[input_idx]
    }
}
