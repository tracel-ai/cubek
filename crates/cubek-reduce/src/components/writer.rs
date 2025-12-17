use crate::{LineMode, ReduceInstruction, ReducePrecision};
use cubecl::{
    prelude::*,
    std::tensor::{
        View,
        layout::{Coords1d, plain::PlainLayout},
        r#virtual::VirtualTensor,
    },
};

#[derive(CubeType)]
/// Abstract how data is written to global memory.
///
/// Depending on the problem kind, writes might be buffered to optimize vectorization, only
/// happening when [Writer::commit()] is called.
pub enum Writer<Out: Numeric> {
    Parallel(ParallelWriter<Out>),
    Perpendicular(PerpendicularWriter<Out>),
}

#[cube]
impl<Out: Numeric> Writer<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        write_index: u32,
        #[comptime] line_mode: LineMode,
    ) -> Writer<Out> {
        match line_mode {
            LineMode::Parallel => Writer::<Out>::new_Parallel(ParallelWriter::<Out>::new::<P>(
                input,
                output,
                reduce_axis,
                write_index,
            )),
            LineMode::Perpendicular => Writer::<Out>::new_Perpendicular(
                PerpendicularWriter::<Out>::new::<P>(input, output, reduce_axis, write_index),
            ),
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: u32,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        match self {
            Writer::Parallel(writer) => writer.write::<P, I>(local_index, accumulator, inst),
            Writer::Perpendicular(writer) => writer.write::<P, I>(local_index, accumulator, inst),
        }
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        match self {
            Writer::Parallel(writer) => writer.commit_required(),
            Writer::Perpendicular(writer) => writer.commit_required(),
        }
    }

    pub fn commit(&mut self) {
        match self {
            Writer::Parallel(writer) => writer.commit(),
            Writer::Perpendicular(writer) => writer.commit(),
        }
    }

    pub fn write_count(&self) -> comptime_type!(u32) {
        match self {
            Writer::Parallel(writer) => writer.write_count(),
            Writer::Perpendicular(writer) => writer.write_count(),
        }
    }
}

#[derive(CubeType)]
pub struct ParallelWriter<Out: Numeric> {
    output: View<Line<Out>, Coords1d, ReadWrite>,
    buffer: Line<Out>,
    axis_size: u32,
    write_index: u32,
}

#[cube]
impl<Out: Numeric> ParallelWriter<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        write_index: u32,
    ) -> ParallelWriter<Out> {
        ParallelWriter::<Out> {
            output: output.view_mut(PlainLayout::new(output.len())),
            buffer: Line::empty(output.line_size()),
            axis_size: input.shape(reduce_axis),
            write_index,
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: u32,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        let line = I::merge_line::<Out>(inst, accumulator, self.axis_size);
        self.buffer[local_index] = line;
    }

    pub fn commit(&mut self) {
        self.output.write(self.write_index, self.buffer)
    }

    pub fn write_count(&self) -> comptime_type!(u32) {
        self.buffer.line_size()
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        true
    }
}

#[derive(CubeType)]
pub struct PerpendicularWriter<Out: Numeric> {
    output: View<Line<Out>, Coords1d, ReadWrite>,
    axis_size: u32,
    #[cube(comptime)]
    input_line_size: u32,
    #[cube(comptime)]
    output_line_size: u32,
    write_index: u32,
}

#[cube]
impl<Out: Numeric> PerpendicularWriter<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        write_index: u32,
    ) -> PerpendicularWriter<Out> {
        let input_line_size = input.line_size();
        let output_line_size = output.line_size();

        PerpendicularWriter::<Out> {
            output: output.view_mut(PlainLayout::new(output.len())),
            axis_size: input.shape(reduce_axis),
            write_index,
            input_line_size,
            output_line_size,
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        _local_index: u32,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        let out = I::to_output_perpendicular(inst, accumulator, self.axis_size);

        if comptime![self.output_line_size == self.input_line_size] {
            self.output.write(self.write_index, out);
        } else {
            let num_iters = comptime![self.input_line_size / self.output_line_size];

            #[unroll]
            for i in 0..num_iters {
                let mut tmp = Line::empty(self.output_line_size);

                #[unroll]
                for j in 0..self.output_line_size {
                    tmp[j] = out[i * self.output_line_size + j];
                }

                let index = self.write_index * num_iters + i;
                self.output.write(index, tmp);
            }
        }
    }

    pub fn commit(&mut self) {
        // Nothing to do.
    }

    pub fn write_count(&self) -> comptime_type!(u32) {
        1u32
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        false
    }
}
