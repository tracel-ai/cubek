use crate::routines::SharedMemoryBlueprint;
use cubecl::prelude::*;

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct SharedMemoryReader<EA: Float, N: Size> {
    smem: Shared<[Vector<EA, N>]>,
    min_row: isize,
    min_col: isize,
    smem_width: usize,
    channel_groups: usize,
    channel_group: usize,
}

#[cube]
impl<EA: Float, N: Size> SharedMemoryReader<EA, N> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<EI: Float>(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        channel_group: usize,
        input_height: usize,
        input_width: usize,
        min_row: isize,
        min_col: isize,
        #[comptime] vector_size: usize,
        #[comptime] blueprint: SharedMemoryBlueprint,
    ) -> SharedMemoryReader<EA, N> {
        let smem_size = blueprint.smem_width * blueprint.smem_height * blueprint.channel_groups;
        let mut smem = Shared::new_slice(smem_size);
        let cube_dim = CUBE_DIM as usize;

        let mut i = UNIT_POS as usize;
        while i < smem_size {
            let local_c = i % blueprint.channel_groups;
            let local_offset = i / blueprint.channel_groups;

            let (global_y, global_x) = if comptime!(blueprint.smem_height == 1) {
                let flat_start = (min_row * input_width as isize) + min_col;
                let flat_current = flat_start + local_offset as isize;

                (
                    flat_current / input_width as isize,
                    flat_current % input_width as isize,
                )
            } else {
                let local_x = local_offset % blueprint.smem_width;
                let local_y = local_offset / blueprint.smem_width;

                (min_row + local_y as isize, min_col + local_x as isize)
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
            min_row,
            min_col,
            smem_width: blueprint.smem_width,
            channel_groups: blueprint.channel_groups,
            channel_group,
        }
    }

    pub fn read_weighted<EI: Float>(
        &self,
        row: usize,
        col: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let local_row = (row as isize - self.min_row).max(0) as usize;
        let local_col = (col as isize - self.min_col).max(0) as usize;

        let smem_idx = (local_row * self.smem_width * self.channel_groups)
            + (local_col * self.channel_groups)
            + self.channel_group;

        self.smem[smem_idx] * weight
    }
}
