use crate::{
    ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        instructions::{Accumulator, AccumulatorFormat, ReduceWithIndices},
        writers::{Writer, WriterExpand},
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

/// Writes a reduction's values *and* their indices from a single pass.
///
/// The two outputs have the same shape and the same reduce/vec axes, so this is
/// just a pair of [`Writer`]s over different tensors: all the layout, buffering
/// and commit logic is reused unchanged, and the only thing added here is
/// splitting one [`ReduceWithIndices`] conversion across the two of them.
///
/// Kept separate from [`Writer`] rather than folded into it because emitting
/// indices needs the stronger [`ReduceWithIndices`] bound, which the values-only
/// path (`Sum`, `Mean`, ...) cannot satisfy.
#[derive(CubeType)]
pub struct IndicesWriter<'a, Out: NumericVector, Idx: NumericVector> {
    values: Writer<'a, Out>,
    indices: Writer<'a, Idx>,
}

#[cube]
impl<'a, Out: NumericVector, Idx: NumericVector> IndicesWriter<'a, Out, Idx> {
    #[allow(clippy::too_many_arguments)]
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        values: &'a mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        indices: &'a mut VirtualTensor<Idx::T, Idx::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        write_index: usize,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] acc_format: AccumulatorFormat,
    ) -> IndicesWriter<'a, Out, Idx> {
        IndicesWriter::<'a, Out, Idx> {
            values: Writer::<Out>::new::<P>(
                input,
                values,
                reduce_axis,
                out_vec_axis,
                write_index,
                vectorization_mode,
                acc_format,
            ),
            indices: Writer::<Idx>::new::<P>(
                input,
                indices,
                reduce_axis,
                out_vec_axis,
                write_index,
                vectorization_mode,
                acc_format,
            ),
        }
    }

    /// Convert the accumulator once and buffer each half into its own writer.
    pub fn write<P: ReducePrecision, I: ReduceWithIndices<P>>(
        &mut self,
        local_index: usize,
        accumulator: Accumulator<P>,
        inst: &I,
    ) {
        match &mut self.values {
            Writer::Parallel(values) => {
                let (out_values, out_indices) = I::to_output_both_parallel::<Out::T, Idx::T>(
                    inst,
                    accumulator,
                    values.axis_size,
                );
                values.push(local_index, out_values);

                match &mut self.indices {
                    Writer::Parallel(indices) => indices.push(local_index, out_indices),
                    Writer::Perpendicular(_) => comptime!(unreachable!(
                        "values and indices writers are built from the same vectorization mode"
                    )),
                }
            }
            Writer::Perpendicular(values) => {
                let (out_values, out_indices) = I::to_output_both_perpendicular::<Out::T, Idx::T>(
                    inst,
                    accumulator,
                    values.axis_size,
                );
                values.push::<P::SI>(out_values);

                match &mut self.indices {
                    Writer::Perpendicular(indices) => indices.push::<P::SI>(out_indices),
                    Writer::Parallel(_) => comptime!(unreachable!(
                        "values and indices writers are built from the same vectorization mode"
                    )),
                }
            }
        }
    }

    pub fn commit_required(&self) -> comptime_type!(bool) {
        self.values.commit_required()
    }

    pub fn commit(&mut self) {
        self.values.commit();
        self.indices.commit();
    }

    pub fn write_count(&self) -> comptime_type!(VectorSize) {
        self.values.write_count()
    }
}
