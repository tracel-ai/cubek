use crate::routines::SharedMemoryBlueprint;
use cubecl::prelude::*;

#[derive(CubeType, Clone)]
#[expand(derive(Clone))]
pub struct SharedMemoryReader<EI: Float, N: Size> {
    smem: Shared<[Vector<EI, N>]>,
    min_row: isize,
    min_col: isize,
    smem_width: usize,
    num_vectors: usize,
    vector_index: usize,
}

#[cube]
impl<EI: Float, N: Size> SharedMemoryReader<EI, N> {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input: &Tensor<Vector<EI, N>>,
        batch: usize,
        vector_index: usize,
        input_height: usize,
        input_width: usize,
        min_row: isize,
        min_col: isize,
        #[comptime] vector_size: usize,
        #[comptime] blueprint: SharedMemoryBlueprint,
    ) -> SharedMemoryReader<EI, N> {
        let smem_size = blueprint.smem_width * blueprint.smem_height * blueprint.num_vectors;
        let mut smem = Shared::new_slice(smem_size);

        let unit_pos = UNIT_POS as usize;
        let cube_dim = CUBE_DIM as usize;

        let num_iterations = (smem_size - unit_pos).div_ceil(cube_dim);

        for step in 0..num_iterations {
            let i = unit_pos + step * cube_dim;

            let local_vector_index = i % blueprint.num_vectors;
            let local_pos = i / blueprint.num_vectors;
            let local_col = local_pos % blueprint.smem_width;
            let local_row = local_pos / blueprint.smem_width;

            let (global_row, global_col) =
                (min_row + local_row as isize, min_col + local_col as isize);

            let input_idx = (batch * input.stride(0)
                + global_row.max(0).min((input_height - 1) as isize) as usize * input.stride(1)
                + global_col.max(0).min((input_width - 1) as isize) as usize * input.stride(2)
                + local_vector_index * input.stride(3) * vector_size)
                / vector_size;

            smem[i] = input[input_idx];
        }

        sync_cube();

        SharedMemoryReader::<EI, N> {
            smem,
            min_row,
            min_col,
            smem_width: blueprint.smem_width,
            num_vectors: blueprint.num_vectors,
            vector_index,
        }
    }

    pub fn read_weighted<EA: Float>(
        &self,
        row: usize,
        col: usize,
        weight: Vector<EA, N>,
    ) -> Vector<EA, N> {
        let local_row = (row as isize - self.min_row).max(0) as usize;
        let local_col = (col as isize - self.min_col).max(0) as usize;

        let smem_idx =
            (local_row * self.smem_width + local_col) * self.num_vectors + self.vector_index;

        Vector::cast_from(self.smem[smem_idx]) * weight
    }
}
