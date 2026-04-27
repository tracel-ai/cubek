use crate::{
    ReduceInstruction, ReducePrecision,
    components::{
        args::NumericVector,
        instructions::{Accumulator, AccumulatorFormat, Value, ValueExpand},
        writers::build_reduce_output_layout,
    },
};
use cubecl::{
    prelude::*,
    std::tensor::{View, layout::Coords2d, r#virtual::VirtualTensor},
};

#[derive(CubeType)]
pub struct ParallelWriter<Out: NumericVector> {
    output: View<Vector<Out::T, Out::N>, Coords2d, ReadWrite>,
    buffer: Value<Vector<Out::T, Out::N>>,
    axis_size: usize,
    write_index: usize,
    #[cube(comptime)]
    accumulator_length: usize,
}

#[cube]
impl<Out: NumericVector> ParallelWriter<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        write_index: usize,
        #[comptime] accumulator_format: AccumulatorFormat,
    ) -> ParallelWriter<Out> {
        let layout = build_reduce_output_layout::<Out>(
            output,
            reduce_axis,
            out_vec_axis,
            accumulator_format.len(),
        );

        ParallelWriter::<Out> {
            output: output.view_mut(layout),
            buffer: match accumulator_format {
                AccumulatorFormat::Single => Value::new_single(Vector::empty()),
                AccumulatorFormat::Multiple(length) => Value::new_Multiple(Array::new(length)),
            },
            axis_size: input.shape(reduce_axis),
            write_index,
            accumulator_length: accumulator_format.len(),
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: usize,
        accumulator: Accumulator<P>,
        inst: &I,
    ) {
        let out = I::to_output_parallel::<Out::T>(inst, accumulator, self.axis_size);

        match out {
            Value::Multiple(array) =>
            {
                #[unroll]
                for i in 0..self.accumulator_length {
                    let mut vec = self.buffer.multiple_mut()[i];
                    vec[local_index] = array[i];
                    self.buffer.multiple_mut()[i] = vec;
                }
            }
            Value::Single(element) => {
                self.buffer.item()[local_index] = element.unwrap();
            }
            Value::None => {
                unreachable!()
            }
        }
    }

    pub fn commit(&mut self) {
        match &mut self.buffer {
            Value::Multiple(array) => {
                let write_index = self.write_index as u32;
                #[unroll]
                for k_iter in 0..self.accumulator_length {
                    let k_u32 = comptime!(k_iter as u32);
                    self.output
                        .write((write_index, k_u32.runtime()), array[k_iter])
                }
            }
            Value::Single(vector) => self
                .output
                .write((self.write_index as u32, 0), vector.unwrap()),
            Value::None => unreachable!(),
        }
    }

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        match &self.buffer {
            Value::Multiple(array) => array[0].vector_size(),
            Value::Single(vector) => vector.unwrap().vector_size(),
            Value::None => unreachable!(),
        }
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        true
    }
}
