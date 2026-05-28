use cubecl::prelude::*;

#[derive(CubeType)]
pub struct Writer {}

#[cube]
impl Writer {
    pub fn write<EI: Float, N: Size>(
        output: &mut Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        row: usize,
        col: usize,
        vector_size: usize,
        value: Vector<EI, N>,
    ) {
        let out_index =
            (batch * output.stride(0) + row * output.stride(1) + col * output.stride(2))
                / vector_size
                + channel_group * output.stride(3);

        output[out_index] = value;
    }
}
