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

    pub fn read_weighted<EI: Float, EA: Float, N: Size>(
        &self,
        input: &Tensor<Vector<EI, N>>,
        row_offset: usize,
        column_offset: usize,
        vector_size: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let input_index = (row_offset + column_offset * input.stride(2)) / vector_size
            + self.channel_group * input.stride(3);

        let pixel = input[input_index];
        Vector::cast_from(pixel) * weight
    }
}
