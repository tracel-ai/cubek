use cubecl::prelude::*;

#[derive(CubeType)]
pub struct Writer {}

#[cube]
impl Writer {
    pub fn write<EI: Float, N: Size>(
        output: &mut Tensor<Vector<EI, N>>,
        batch: usize,
        vector_index: usize,
        row: usize,
        col: usize,
        vector_size: usize,
        value: Vector<EI, N>,
    ) {
        let out_index =
            (batch * output.stride(0) + row * output.stride(1) + col * output.stride(2))
                / vector_size
                + vector_index * output.stride(3);

        output[out_index] = value;
    }
}
