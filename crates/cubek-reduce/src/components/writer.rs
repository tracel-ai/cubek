use crate::{LineMode, ReduceInstruction, ReducePrecision};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct ParallelWriter<Out: Numeric> {
    buffer: Line<Out>,
    axis_size: u32,
}

#[derive(CubeType)]
pub struct PerpendicularWriter<Out: Numeric> {
    buffer: Array<Line<Out>>,
    axis_size: u32,
    #[cube(comptime)]
    input_line_size: u32,
    #[cube(comptime)]
    output_line_size: u32,
}

#[derive(CubeType)]
pub enum Writer<Out: Numeric> {
    Parallel(ParallelWriter<Out>),
    PerpendicularWriter(PerpendicularWriter<Out>),
}

#[cube]
impl<Out: Numeric> Writer<Out> {
    pub fn new(
        axis_size: u32,
        #[comptime] input_line_size: u32,
        #[comptime] output_line_size: u32,
        #[comptime] line_mode: LineMode,
    ) -> Writer<Out> {
        match line_mode {
            LineMode::Parallel => {
                Writer::<Out>::new_Parallel(ParallelWriter::<Out>::new(axis_size, output_line_size))
            }
            LineMode::Perpendicular => Writer::<Out>::new_PerpendicularWriter(
                PerpendicularWriter::<Out>::new(axis_size, input_line_size, output_line_size),
            ),
        }
    }

    pub fn accumulate<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: u32,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        match self {
            Writer::Parallel(writer) => writer.accumulate::<P, I>(local_index, accumulator, inst),
            Writer::PerpendicularWriter(writer) => {
                writer.accumulate::<P, I>(local_index, accumulator, inst)
            }
        }
    }

    pub fn commit(&mut self, output: &mut VirtualTensor<Out, ReadWrite>, start_index: u32) {
        match self {
            Writer::Parallel(writer) => writer.commit(output, start_index),
            Writer::PerpendicularWriter(writer) => writer.commit(output, start_index),
        }
    }

    pub fn num_accumulate(&self) -> comptime_type!(u32) {
        match self {
            Writer::Parallel(writer) => writer.num_accumulate(),
            Writer::PerpendicularWriter(writer) => writer.num_accumulate(),
        }
    }
}

#[cube]
impl<Out: Numeric> ParallelWriter<Out> {
    pub fn new(axis_size: u32, #[comptime] output_line_size: u32) -> ParallelWriter<Out> {
        ParallelWriter::<Out> {
            buffer: Line::empty(output_line_size),
            axis_size,
        }
    }

    pub fn accumulate<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: u32,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        let line = I::merge_line::<Out>(inst, accumulator, self.axis_size);
        self.buffer[local_index] = line;
    }

    pub fn commit(&mut self, output: &mut VirtualTensor<Out, ReadWrite>, reduce_index: u32) {
        output.write(reduce_index, self.buffer)
    }

    pub fn num_accumulate(&self) -> comptime_type!(u32) {
        self.buffer.line_size()
    }
}

#[cube]
impl<Out: Numeric> PerpendicularWriter<Out> {
    pub fn new(
        axis_size: u32,
        #[comptime] input_line_size: u32,
        #[comptime] output_line_size: u32,
    ) -> PerpendicularWriter<Out> {
        let buffer = if comptime![output_line_size == input_line_size] {
            Array::vectorized(1u32, output_line_size)
        } else {
            let length = comptime![input_line_size / output_line_size];
            Array::vectorized(length, output_line_size)
        };
        PerpendicularWriter::<Out> {
            buffer,
            axis_size,
            input_line_size,
            output_line_size,
        }
    }

    pub fn accumulate<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        _local_index: u32,
        accumulator: I::AccumulatorItem,
        inst: &I,
    ) {
        let out = I::to_output_perpendicular(inst, accumulator, self.axis_size);

        if comptime![self.output_line_size == self.input_line_size] {
            self.buffer[0u32] = out;
        } else {
            let num_iters = comptime![self.input_line_size / self.output_line_size];

            #[unroll]
            for i in 0..num_iters {
                let mut tmp = Line::empty(self.output_line_size);

                #[unroll]
                for j in 0..self.output_line_size {
                    tmp[j] = out[i * self.output_line_size + j];
                }

                self.buffer[i] = tmp;
            }
        }
    }

    pub fn commit(&mut self, output: &mut VirtualTensor<Out, ReadWrite>, reduce_index: u32) {
        if comptime![self.output_line_size == self.input_line_size] {
            output.write(reduce_index, self.buffer[0]);
        } else {
            let num_iters = comptime![self.input_line_size / self.output_line_size];

            #[unroll]
            for i in 0..num_iters {
                let index = reduce_index * num_iters + i;
                output.write(index, self.buffer[i]);
            }
        }
    }

    pub fn num_accumulate(&self) -> comptime_type!(u32) {
        1u32
    }
}

#[cube]
pub fn write_accumulator<P: ReducePrecision, Out: Numeric, R: ReduceInstruction<P>>(
    output: &mut VirtualTensor<Out, ReadWrite>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] input_line_size: u32,
    inst: &R,
) {
    match comptime!(line_mode) {
        LineMode::Parallel => {
            let result = R::merge_line::<Out>(inst, accumulator, shape_axis_reduce);
            output.write(reduce_index, Line::cast_from(result))
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(inst, accumulator, shape_axis_reduce);
            let output_line_size = output.line_size();

            if comptime![output_line_size == input_line_size] {
                output.write(reduce_index, out);
            } else {
                let num_iters = comptime![input_line_size / output_line_size];

                #[unroll]
                for i in 0..num_iters {
                    let mut tmp = Line::empty(output_line_size);

                    #[unroll]
                    for j in 0..output_line_size {
                        tmp[j] = out[i * output_line_size + j];
                    }

                    let index = num_iters * reduce_index + i;
                    output.write(index, tmp);
                }
            }
        }
    }
}
