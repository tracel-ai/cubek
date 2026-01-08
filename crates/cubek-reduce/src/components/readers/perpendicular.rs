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
pub struct PerpendicularReader<P: ReducePrecision> {
    view: View<Line<P::EI>, Coords1d>,
    /// The global offset that points where the vector to reduce is located in global memory.
    batch_offset: usize,
    vector_offset_stride: usize,
    requirements: ReduceRequirements,
    #[cube(comptime)]
    line_size: LineSize,
    bound_checks: ReaderBoundChecks<P>,
    shape: usize,
}

#[cube]
impl<P: ReducePrecision> PerpendicularReader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: usize,
        reduce_index: usize,
        idle: CubeOption<bool>,
        #[comptime] bound_checks: BoundChecks,
    ) -> PerpendicularReader<P> {
        let line_size = input.line_size();
        let output_index = reduce_index * line_size;

        let mut batch_offset = 0;
        for axis in 0..input.rank() {
            let coordinate = output.coordinate(output_index, axis);
            batch_offset += coordinate * input.stride(axis);
        }
        batch_offset /= line_size;

        let requirements = I::requirements(inst);
        let vector_offset_stride = input.stride(reduce_axis) / line_size;
        let shape = input.shape(reduce_axis);

        let bound_checks = ReaderBoundChecks::new::<I>(inst, shape, idle, line_size, bound_checks);

        PerpendicularReader::<P> {
            view: input.view(PlainLayout::new(input.len())),
            batch_offset,
            vector_offset_stride,
            requirements,
            line_size,
            bound_checks,
            shape,
        }
    }

    pub fn length_unit(&self) -> usize {
        self.shape
    }

    pub fn length_plane(&self) -> usize {
        self.shape.div_ceil(CUBE_DIM_X as usize)
    }

    pub fn length_cube(&self) -> usize {
        self.shape.div_ceil(CUBE_DIM as usize)
    }

    pub fn read_cube(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_pos = line_index * CUBE_DIM as usize;
        let unit_pos = UNIT_POS as usize;
        let pos = plane_pos + unit_pos;
        let offset = plane_pos * self.vector_offset_stride
            + unit_pos * self.vector_offset_stride
            + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            plane_pos + unit_pos,
            self.requirements,
            self.line_size,
            LineMode::Perpendicular,
        );

        (item, coordinate)
    }

    pub fn read_plane(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
        let plane_pos = line_index * CUBE_DIM_X as usize;
        let unit_pos = UNIT_POS_X as usize;
        let pos = plane_pos + unit_pos;
        let offset = plane_pos * self.vector_offset_stride
            + unit_pos * self.vector_offset_stride
            + self.batch_offset;

        let item = self.bound_checks.read(pos, offset, &self.view);

        let coordinate = ReduceCoordinate::new(
            plane_pos + unit_pos,
            self.requirements,
            self.line_size,
            LineMode::Perpendicular,
        );

        (item, coordinate)
    }

    pub fn read_unit(&self, line_index: usize) -> (Line<P::EI>, ReduceCoordinate) {
        let offset = self.batch_offset + line_index * self.vector_offset_stride;
        let item = self.view[offset];

        let coordinate = ReduceCoordinate::new(
            line_index,
            self.requirements,
            self.line_size,
            LineMode::Perpendicular,
        );

        (item, coordinate)
    }
}
