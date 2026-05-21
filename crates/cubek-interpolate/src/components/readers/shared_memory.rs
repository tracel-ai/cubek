use crate::components::readers::{Reader, ReaderExpand};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryReader<EA: Float, N: Size> {
    smem: SharedMemory<Vector<EA, N>>,
    min_x: usize,
    min_y: usize,
    smem_width: usize,
}

#[cube]
impl<EA: Float, N: Size> Reader<EA, N> for SharedMemoryReader<EA, N> {
    fn prepare_read<EI: Float>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        input_width: usize,
        input_height: usize,
        min_x: usize,
        min_y: usize,
        #[comptime] vector_size: usize,
        #[comptime] smem_width: usize,
        #[comptime] smem_height: usize,
    ) -> SharedMemoryReader<EA, N> {
        let smem_size = (smem_width * smem_height) / vector_size;

        let mut smem = SharedMemory::<Vector<EA, N>>::new(smem_size);

        let cube_dim = CUBE_DIM as usize;
        let base_offset = batch * input.stride(0) + channel_group * input.stride(3) * vector_size;

        let mut i = UNIT_POS as usize;
        while i < smem_size {
            let local_y = i / smem_width;
            let local_x = i % smem_width;

            let global_y = min_y + local_y;
            let global_x = min_x + local_x;

            let global_idx = (base_offset
                + global_y.max(0).min(input_height - 1) * input.stride(1)
                + global_x.max(0).min(input_width - 1) * input.stride(2))
                / vector_size;

            smem[i] = Vector::cast_from(input[global_idx]);
            i += cube_dim;
        }

        sync_cube();

        SharedMemoryReader::<EA, N> {
            smem,
            min_x,
            min_y,
            smem_width,
        }
    }

    fn read_weighted<EI: Float>(
        &self,
        _input: &Tensor<Vector<EI, N>>,
        y: usize,
        x: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let local_x = x - self.min_x;
        let local_y = y - self.min_y;
        let smem_idx = local_y * self.smem_width + local_x;

        self.smem[smem_idx] * weight
    }
}
