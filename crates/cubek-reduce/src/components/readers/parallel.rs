use crate::{
    BoundChecks, LineMode, ReduceInstruction, ReducePrecision,
    components::{
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
    view: View<Line<P::EI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    batch_offset: u32,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    line_size: u32,
    bound_checks: ReaderBoundChecks<P>,
    num_chunks: u32,
}

#[cube]
impl<P: ReducePrecision> ParallelReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
        #[comptime] bound_checks: BoundChecks,
    ) -> ParallelReader<P> {
        let line_size = input.line_size();

        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(reduce_index, axis);
            batch_offset += coordinate * input.stride(axis);
        }
        batch_offset /= line_size;

        let requirements = I::requirements(inst);

        let shape = input.shape(reduce_axis);

        let num_chunks = shape / line_size;
        let bound_checks = ReaderBoundChecks::new::<I>(inst, num_chunks, line_size, bound_checks);

        ParallelReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            batch_offset,
            requirements,
            line_size,
            bound_checks,
            num_chunks,
        }
    }

    pub fn length_unit(&self) -> u32 {
        self.num_chunks
    }

    pub fn length_plane(&self) -> u32 {
        self.num_chunks.div_ceil(CUBE_DIM_X)
    }

    pub fn length_cube(&self) -> u32 {
        self.num_chunks.div_ceil(CUBE_DIM)
    }

    pub fn read_cube(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_pos = line_index * CUBE_DIM;
        let unit_pos = UNIT_POS;
        let pos = plane_pos + unit_pos;
        let offset = pos + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            (line_index * self.line_size * CUBE_DIM) + UNIT_POS * self.line_size,
            self.requirements,
            self.line_size,
            LineMode::Parallel,
        );

        (item, coordinate)
    }

    pub fn read_plane(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_pos = line_index * CUBE_DIM_X;
        let unit_pos = UNIT_POS_X;
        let pos = plane_pos + unit_pos;
        let offset = pos + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            (line_index * self.line_size * CUBE_DIM_X) + UNIT_POS_X * self.line_size,
            self.requirements,
            self.line_size,
            LineMode::Parallel,
        );

        (item, coordinate)
    }

    pub fn read_unit(&self, line_index: u32) -> (Line<P::EI>, ReduceCoordinate) {
        let offset = line_index + self.batch_offset;
        let item = self.view[offset];

        let coordinate = ReduceCoordinate::new(
            line_index * self.line_size,
            self.requirements,
            self.line_size,
            LineMode::Parallel,
        );

        (item, coordinate)
    }
}
