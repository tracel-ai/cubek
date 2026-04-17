use crate::{
    BoundChecks, ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericLine,
        instructions::{ReduceCoordinate, ReduceRequirements},
        readers::bound_checks::ReaderBoundChecks,
    },
};
use cubecl::{
    prelude::*,
    std::tensor::{
        View,
        layout::{Coords1d, plain::PlainLayout},
        r#virtual::VirtualTensor,
    },
};

#[derive(CubeType)]
pub struct ParallelReader<P: ReducePrecision> {
    view: View<Vector<P::EI, P::SI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    batch_offset: usize,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    vector_size: VectorSize,
    bound_checks: ReaderBoundChecks<P>,
    num_chunks: usize,
    effective_plane_dim: u32,
}

#[cube]
impl<P: ReducePrecision> ParallelReader<P> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<I: ReduceInstruction<P>, Out: NumericLine>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        inst: &I,
        reduce_axis: usize,
        reduce_index: usize,
        idle: ComptimeOption<bool>,
        effective_plane_dim: u32,
        #[comptime] bound_checks: BoundChecks,
    ) -> ParallelReader<P> {
        let vector_size = input.vector_size();

        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            batch_offset += coordinate * input.stride(axis);
        }
        batch_offset /= vector_size;

        let requirements = I::requirements(inst);

        let shape = input.shape(reduce_axis);

        let num_chunks = shape / vector_size;
        let bound_checks = ReaderBoundChecks::new::<I>(inst, num_chunks, idle, bound_checks);

        ParallelReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            batch_offset,
            requirements,
            vector_size,
            bound_checks,
            num_chunks,
            effective_plane_dim,
        }
    }

    pub fn length_unit(&self) -> usize {
        self.num_chunks
    }

    pub fn length_plane(&self) -> usize {
        self.num_chunks.div_ceil(self.effective_plane_dim as usize)
    }

    pub fn length_cube(&self) -> usize {
        self.num_chunks.div_ceil(CUBE_DIM as usize)
    }

    pub fn read_cube(
        &self,
        vector_index: usize,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        let cube_dim = CUBE_DIM as usize;
        let plane_pos = vector_index * cube_dim;
        let unit_pos = UNIT_POS as usize;
        let pos = plane_pos + unit_pos;
        let offset = pos + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            (plane_pos * self.vector_size) + unit_pos * self.vector_size,
            self.requirements,
            VectorizationMode::Parallel,
        );

        (item, coordinate)
    }

    pub fn read_plane(
        &self,
        vector_index: usize,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        let plane_pos = vector_index * self.effective_plane_dim as usize;
        let unit_pos = UNIT_POS_X as usize;
        let pos = plane_pos + unit_pos;
        let offset = pos + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            (plane_pos * self.vector_size) + unit_pos * self.vector_size,
            self.requirements,
            VectorizationMode::Parallel,
        );

        (item, coordinate)
    }

    pub fn read_unit(
        &self,
        vector_index: usize,
    ) -> (Vector<P::EI, P::SI>, ReduceCoordinate<P::SI>) {
        let offset = vector_index + self.batch_offset;
        let item = self.view[offset];

        let coordinate = ReduceCoordinate::new(
            vector_index * self.vector_size,
            self.requirements,
            VectorizationMode::Parallel,
        );

        (item, coordinate)
    }
}
