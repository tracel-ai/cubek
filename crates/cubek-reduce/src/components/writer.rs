use crate::{ReduceInstruction, ReducePrecision, VectorizationMode, components::args::NumericLine};
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
pub enum Writer<Out: NumericLine> {
    Parallel(ParallelWriter<Out>),
    Perpendicular(PerpendicularWriter<Out>),
}

#[cube]
impl<Out: NumericLine> Writer<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        write_index: usize,
        #[comptime] vectorization_mode: VectorizationMode,
    ) -> Writer<Out> {
        match vectorization_mode {
            VectorizationMode::Parallel => Writer::<Out>::new_Parallel(
                ParallelWriter::<Out>::new::<P>(input, output, reduce_axis, write_index),
            ),
            VectorizationMode::Perpendicular => Writer::<Out>::new_Perpendicular(
                PerpendicularWriter::<Out>::new::<P>(input, output, reduce_axis, write_index),
            ),
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: usize,
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

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        match self {
            Writer::Parallel(writer) => writer.write_count(),
            Writer::Perpendicular(writer) => writer.write_count(),
        }
    }
}

#[derive(CubeType)]
pub struct ParallelWriter<Out: NumericLine> {
    output: View<Vector<Out::T, Out::N>, Coords1d, ReadWrite>,
    buffer: Vector<Out::T, Out::N>,
    axis_size: usize,
    write_index: usize,
}

#[cube]
impl<Out: NumericLine> ParallelWriter<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        write_index: usize,
    ) -> ParallelWriter<Out> {
        ParallelWriter::<Out> {
            output: output.view_mut(PlainLayout::new(output.len())),
            buffer: Vector::empty(),
            axis_size: input.shape(reduce_axis),
            write_index,
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: usize,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        let vector = I::merge_vector::<Out::T>(inst, accumulator, self.axis_size);
        self.buffer[local_index] = vector;
    }

    pub fn commit(&mut self) {
        self.output.write(self.write_index, self.buffer)
    }

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        self.buffer.vector_size()
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        true
    }
}

#[derive(CubeType)]
pub struct PerpendicularWriter<Out: NumericLine> {
    output: View<Vector<Out::T, Out::N>, Coords1d, ReadWrite>,
    axis_size: usize,
    #[cube(comptime)]
    input_vector_size: VectorSize,
    #[cube(comptime)]
    output_vector_size: VectorSize,
    write_index: usize,
}

#[cube]
impl<Out: NumericLine> PerpendicularWriter<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        write_index: usize,
    ) -> PerpendicularWriter<Out> {
        let input_vector_size = input.vector_size();
        let output_vector_size = output.vector_size();

        PerpendicularWriter::<Out> {
            output: output.view_mut(PlainLayout::new(output.len())),
            axis_size: input.shape(reduce_axis),
            write_index,
            input_vector_size,
            output_vector_size,
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        _local_index: usize,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        let out = I::to_output_perpendicular::<Out::T>(inst, accumulator, self.axis_size);

        if comptime![self.output_vector_size == self.input_vector_size] {
            self.output.write(self.write_index, Vector::cast_from(out));
        } else {
            let num_iters = comptime![self.input_vector_size / self.output_vector_size];

            #[unroll]
            for i in 0..num_iters {
                let mut tmp = Vector::empty();

                #[unroll]
                for j in 0..self.output_vector_size {
                    tmp[j] = out[i * self.output_vector_size + j];
                }

                let index = self.write_index * num_iters + i;
                self.output.write(index, tmp);
            }
        }
    }

    pub fn commit(&mut self) {
        // Nothing to do.
    }

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        1
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        false
    }
}
