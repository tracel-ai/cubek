use cubecl::prelude::*;

#[derive(CubeType)]
pub struct Writer {}

#[cube]
impl Writer {
    pub fn new() -> Writer {
        Writer {}
    }

    pub fn write<EI: Float, N: Size>(
        &self,
        output: &mut Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        x: usize,
        y: usize,
        vector_size: usize,
        value: Vector<EI, N>,
    ) {
        let out_index = (batch * output.stride(0) + y * output.stride(1) + x * output.stride(2))
            / vector_size
            + channel_group * output.stride(3);

        output[out_index] = value;
    }
}
