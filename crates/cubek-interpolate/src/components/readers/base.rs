use cubecl::prelude::*;

#[cube]
pub trait Reader<EA: Float, N: Size>: 'static {
    fn init<EI: Float>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        vector_size: usize,
        input_width: usize,
        input_height: usize,
        min_x: usize,
        min_y: usize,
        #[comptime] smem_width: usize,
        #[comptime] smem_height: usize,
    ) -> Self;

    fn read_weighted<EI: Float>(
        &self,
        input: &Tensor<Vector<EI, N>>,
        y: usize,
        x: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N>;
}
