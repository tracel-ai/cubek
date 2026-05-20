use cubecl::prelude::*;

#[derive(CubeType)]
pub struct Reader {
    channel_group: usize,
}

#[cube]
impl Reader {
    pub fn new(channel_group: usize) -> Self {
        Reader { channel_group }
    }

    pub fn read_weighted<F: Float, N: Size>(
        &self,
        input: &Tensor<Vector<F, N>>,
        row_offset: usize,
        column_offset: usize,
        vector_size: usize,
        weight: Vector<F, N>,
    ) -> Vector<F, N> {
        let input_index = (row_offset + column_offset * input.stride(2)) / vector_size
            + self.channel_group * input.stride(3);

        let pixel = input[input_index];
        pixel * weight
    }
}
