use crate::{
    BoundChecks, LineMode, ReduceInstruction, ReducePrecision,
    components::{
        instructions::{ReduceCoordinate, ReduceRequirements},
        readers::bound_checks::ReaderBoundChecks,
    },
};
use cubecl::{
    prelude::*,
    std::{
        CubeOption,
        tensor::{
            View,
            layout::{Coords1d, plain::PlainLayout},
            r#virtual::VirtualTensor,
        },
    },
};

#[derive(CubeType)]
pub struct ParallelReader<P: ReducePrecision> {
    view: View<Line<P::EI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    batch_offset: usize,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    line_size: LineSize,
    bound_checks: ReaderBoundChecks<P>,
    num_chunks: usize,
}

#[cube]
impl<P: ReducePrecision> ParallelReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: usize,
        reduce_index: usize,
        idle: CubeOption<bool>,
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
        let bound_checks =
            ReaderBoundChecks::new::<I>(inst, num_chunks, idle, line_size, bound_checks);

        ParallelReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            batch_offset,
            requirements,
            line_size,
            bound_checks,
            num_chunks,
        }
    }

    pub fn length_unit(&self) -> usize {
        self.num_chunks
    }

    pub fn length_plane(&self) -> usize {
        self.num_chunks.div_ceil(CUBE_DIM_X as usize)
    }

    pub fn length_cube(&self) -> usize {
        self.num_chunks.div_ceil(CUBE_DIM as usize)
    }

    pub fn read_cube(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_pos = line_index * CUBE_DIM as usize;
        let unit_pos = UNIT_POS as usize;
        let pos = plane_pos + unit_pos;
        let offset = pos + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            (plane_pos * self.line_size) + unit_pos * self.line_size,
            self.requirements,
            self.line_size,
            LineMode::Parallel,
        );

        (item, coordinate)
    }

    pub fn read_plane(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_pos = line_index * CUBE_DIM_X as usize;
        let unit_pos = UNIT_POS_X as usize;
        let pos = plane_pos + unit_pos;
        let offset = pos + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            (plane_pos * self.line_size) + unit_pos * self.line_size,
            self.requirements,
            self.line_size,
            LineMode::Parallel,
        );

        (item, coordinate)
    }

    pub fn read_unit(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
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
