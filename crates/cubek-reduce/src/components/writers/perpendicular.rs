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
    std::tensor::{ViewMut, layout::Coords2d, r#virtual::VirtualTensor},
};

#[derive(CubeType)]
pub struct PerpendicularWriter<'a, Out: NumericVector> {
    output: ViewMut<'a, Vector<Out::T, Out::N>, Coords2d>,
    pub(crate) axis_size: usize,
    #[cube(comptime)]
    input_vector_size: VectorSize,
    #[cube(comptime)]
    output_vector_size: VectorSize,
    write_index: usize,
    #[cube(comptime)]
    accumulator_length: usize,
}

#[cube]
impl<'a, Out: NumericVector> PerpendicularWriter<'a, Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &'a mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        write_index: usize,
        #[comptime] accumulator_format: AccumulatorFormat,
    ) -> PerpendicularWriter<'a, Out> {
        let input_vector_size = input.vector_size();
        let output_vector_size = output.vector_size();

        let layout = build_reduce_output_layout::<Out>(
            &*output,
            reduce_axis,
            out_vec_axis,
            accumulator_format.len(),
        );

        PerpendicularWriter::<'a, Out> {
            output: output.view_mut(layout),
            axis_size: input.shape(reduce_axis),
            write_index,
            input_vector_size,
            output_vector_size,
            accumulator_length: accumulator_format.len(),
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        _local_index: usize,
        accumulator: Accumulator<P>,
        inst: &I,
    ) {
        let out = I::to_output_perpendicular::<Out::T>(inst, accumulator, self.axis_size);
        self.push::<P::SI>(out);
    }

    /// Write an already-converted reduction result.
    ///
    /// Split out of [`Self::write`] so that a caller holding both a value and an
    /// index result can write each through its own writer, without the writer
    /// having to know how the results were produced.
    pub fn push<S: Size>(&mut self, out: Value<Vector<Out::T, S>>) {
        match out {
            Value::Multiple(array) => self.write_multiple::<S>(array),
            Value::Single(vector) => self.write_single::<S>(vector.unwrap(), 0),
            Value::None => unreachable!(),
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

#[cube]
impl<Out: NumericVector> PerpendicularWriter<'_, Out> {
    fn write_single<S: Size>(&mut self, vector: Vector<Out::T, S>, k_index: usize) {
        if comptime![self.output_vector_size == self.input_vector_size] {
            self.output.write(
                (self.write_index as u32, k_index as u32),
                Vector::cast_from(vector),
            );
        } else {
            let num_iters = comptime![self.input_vector_size / self.output_vector_size];

            #[unroll]
            for i in 0..num_iters {
                let mut tmp = Vector::empty();

                #[unroll]
                for j in 0..self.output_vector_size {
                    tmp.insert(
                        j,
                        Out::T::cast_from(vector.extract(i * self.output_vector_size + j)),
                    );
                }

                let index = self.write_index * num_iters + i;
                self.output.write((index as u32, k_index as u32), tmp);
            }
        }
    }

    fn write_multiple<S: Size>(&mut self, array: Array<Vector<Out::T, S>>) {
        #[unroll]
        for i in 0..self.accumulator_length {
            self.write_single(array[i], i);
        }
    }
}
