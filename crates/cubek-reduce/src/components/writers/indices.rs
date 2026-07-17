use crate::{
    ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        instructions::{Accumulator, AccumulatorFormat, ReduceWithIndices},
        writers::{ReduceWriter, Writer, WriterExpand},
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

/// Writes a reduction's values *and* their indices from a single pass.
///
/// The two outputs have the same shape and the same reduce/vec axes, so this is
/// just a pair of [`Writer`]s over different tensors sharing all the layout,
/// buffering and commit logic; the only thing added here is splitting one
/// [`ReduceWithIndices`] conversion across the two of them.
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
}

#[cube]
impl<'a, Out: NumericVector, Idx: NumericVector, P: ReducePrecision, I: ReduceWithIndices<P>>
    ReduceWriter<P, I> for IndicesWriter<'a, Out, Idx>
{
    /// Convert the accumulator once and buffer each half into its own writer.
    fn write(this: &mut Self, local_index: usize, accumulator: Accumulator<P>, inst: &I) {
        match &mut this.values {
            Writer::Parallel(values) => {
                let (out_values, out_indices) = I::to_output_both_parallel::<Out::T, Idx::T>(
                    inst,
                    accumulator,
                    values.axis_size,
                );
                values.push(local_index, out_values);

                match &mut this.indices {
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

                match &mut this.indices {
                    Writer::Perpendicular(indices) => indices.push::<P::SI>(out_indices),
                    Writer::Parallel(_) => comptime!(unreachable!(
                        "values and indices writers are built from the same vectorization mode"
                    )),
                }
            }
        }
    }

    fn commit_required(this: &Self) -> comptime_type!(bool) {
        this.values.commit_required()
    }

    fn commit(this: &mut Self) {
        this.values.commit();
        this.indices.commit();
    }

    fn write_count(this: &Self) -> comptime_type!(VectorSize) {
        this.values.write_count()
    }
}
