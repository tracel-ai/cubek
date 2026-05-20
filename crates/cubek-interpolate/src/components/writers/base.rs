use cubecl::prelude::*;

#[derive(CubeType)]
pub struct Writer {
    channel_group: usize,
}

#[cube]
impl Writer {
    pub fn new(channel_group: usize) -> Self {
        Writer { channel_group }
    }

    pub fn write<F: Float, N: Size>(
        &self,
        output: &mut Tensor<Vector<F, N>>,
        batch: usize,
        x: usize,
        y: usize,
        vector_size: usize,
        value: Vector<F, N>,
    ) {
        let out_index = (batch * output.stride(0) + y * output.stride(1) + x * output.stride(2))
            / vector_size
            + self.channel_group * output.stride(3);

        output[out_index] = value;
    }
}
