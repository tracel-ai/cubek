use crate::{definition::TileSize, routines::SharedMemoryBlueprint};
use cubecl::prelude::*;

#[derive(CubeType, Clone, Copy)]
pub struct SharedMemoryReader<EA: Float, N: Size> {
    smem: SharedMemory<Vector<EA, N>>,
    min_x: isize,
    min_y: isize,
    smem_width: usize,
    channel_groups: usize,
    channel_group: usize,
}

#[cube]
impl<EA: Float, N: Size> SharedMemoryReader<EA, N> {
    pub fn new<EI: Float>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        input_width: usize,
        input_height: usize,
        min_x: isize,
        min_y: isize,
        #[comptime] vector_size: usize,
        #[comptime] blueprint: SharedMemoryBlueprint,
    ) -> SharedMemoryReader<EA, N> {
        let smem_size = blueprint.smem_width * blueprint.smem_height * blueprint.channel_groups;
        let mut smem = SharedMemory::<Vector<EA, N>>::new(smem_size);
        let cube_dim = CUBE_DIM as usize;

        let mut i = UNIT_POS as usize;
        while i < smem_size {
            let local_c = i % blueprint.channel_groups;
            let local_offset = i / blueprint.channel_groups;

            let (global_y, global_x) = if comptime!(blueprint.smem_height == 1) {
                let flat_start = (min_y * input_width as isize) + min_x;
                let flat_current = flat_start + local_offset as isize;

                (
                    flat_current / input_width as isize,
                    flat_current % input_width as isize,
                )
            } else {
                let local_x = local_offset % blueprint.smem_width;
                let local_y = local_offset / blueprint.smem_width;

                (min_y + local_y as isize, min_x + local_x as isize)
            };

            let global_idx = (batch * input.stride(0)
                + local_c * input.stride(3) * vector_size
                + global_y.max(0).min(input_height.saturating_sub(1) as isize) as usize
                    * input.stride(1)
                + global_x.max(0).min(input_width.saturating_sub(1) as isize) as usize
                    * input.stride(2))
                / vector_size;

            smem[i] = Vector::cast_from(input[global_idx]);
            i += cube_dim;
        }

        sync_cube();

        SharedMemoryReader::<EA, N> {
            smem,
            min_x,
            min_y,
            smem_width: blueprint.smem_width,
            channel_groups: blueprint.channel_groups,
            channel_group,
        }
    }

    pub fn read_weighted<EI: Float>(
        &self,
        x: usize,
        y: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let local_x = (x as isize - self.min_x).max(0) as usize;
        let local_y = (y as isize - self.min_y).max(0) as usize;

        let smem_idx = (local_y * self.smem_width * self.channel_groups)
            + (local_x * self.channel_groups)
            + self.channel_group;

        self.smem[smem_idx] * weight
    }
}
