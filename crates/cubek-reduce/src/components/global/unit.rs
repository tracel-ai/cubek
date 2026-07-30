use crate::{
    BoundChecks, ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        global::{idle_check, reduction_output_base},
        instructions::{Accumulator, ReduceStep, ReduceWithIndices, reduce_inplace},
        readers::{Reader, unit::UnitReader},
        writers::{IndicesWriter, ReduceWriter, Writer},
    },
    routines::UnitReduceBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullUnitReduce;

#[cube]
impl GlobalFullUnitReduce {
    pub fn execute<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        inst: &I,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: UnitReduceBlueprint,
    ) {
        let acc_format = I::accumulator_format(inst);
        let write_index = reduction_output_base::<Out::T, Out::N>(
            ABSOLUTE_POS,
            &*output,
            reduce_axis,
            comptime!(acc_format.len()),
        );

        let mut out = output.clone();
        let mut writer = Writer::<Out>::new::<P>(
            input,
            &mut out,
            reduce_axis,
            out_vec_axis,
            write_index,
            vectorization_mode,
            acc_format,
        );

        Self::reduce_to_writer::<P, Out, I, Writer<Out>>(
            input,
            output,
            reduce_axis,
            write_index,
            inst,
            &mut writer,
            vectorization_mode,
            blueprint,
        );
    }

    /// Same reduction as [`Self::execute`], but writing the values and their
    /// indices to two outputs from a single pass. `indices` must have the same
    /// shape and the same reduce/vec axes as `output`.
    #[allow(clippy::too_many_arguments)]
    pub fn execute_with_indices<
        P: ReducePrecision,
        Out: NumericVector,
        Idx: NumericVector,
        I: ReduceWithIndices<P>,
    >(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        indices: &mut VirtualTensor<Idx::T, Idx::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        inst: &I,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: UnitReduceBlueprint,
    ) {
        let acc_format = I::accumulator_format(inst);
        let write_index = reduction_output_base::<Out::T, Out::N>(
            ABSOLUTE_POS,
            &*output,
            reduce_axis,
            comptime!(acc_format.len()),
        );

        let mut out = output.clone();
        let mut idx = indices.clone();
        let mut writer = IndicesWriter::<Out, Idx>::new::<P>(
            input,
            &mut out,
            &mut idx,
            reduce_axis,
            out_vec_axis,
            write_index,
            vectorization_mode,
            acc_format,
        );

        Self::reduce_to_writer::<P, Out, I, IndicesWriter<Out, Idx>>(
            input,
            output,
            reduce_axis,
            write_index,
            inst,
            &mut writer,
            vectorization_mode,
            blueprint,
        );
    }

    /// The reduction body shared by [`Self::execute`] and
    /// [`Self::execute_with_indices`], generic over how results are written.
    #[allow(clippy::too_many_arguments)]
    fn reduce_to_writer<
        P: ReducePrecision,
        Out: NumericVector,
        I: ReduceInstruction<P>,
        W: ReduceWriter<P, I>,
    >(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        write_index: usize,
        inst: &I,
        writer: &mut W,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] blueprint: UnitReduceBlueprint,
    ) {
        let write_count = W::write_count(&*writer);
        let reduce_index_start = write_index * write_count;

        let idle = idle_check::<P, Out>(
            input,
            &*output,
            reduce_index_start,
            vectorization_mode,
            blueprint.unit_idle,
        );

        for b in 0..write_count {
            let reduce_index = reduce_index_start + b;
            let accumulator = Self::reduce_single::<P, Out, I>(
                input,
                output,
                reduce_axis,
                reduce_index,
                inst,
                idle,
                vectorization_mode,
            );
            W::write(writer, b, accumulator, inst);
        }

        W::commit(writer);
    }

    #[allow(clippy::too_many_arguments)]
    pub fn reduce_single<P: ReducePrecision, Out: NumericVector, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        reduce_index: usize,
        inst: &I,
        idle: ComptimeOption<bool>,
        #[comptime] vectorization_mode: VectorizationMode,
    ) -> Accumulator<P> {
        let reader = Reader::<P>::new::<I, Out>(
            input,
            output,
            inst,
            reduce_axis,
            reduce_index,
            idle,
            comptime!(BoundChecks::None),
            vectorization_mode,
            false,
        );
        let reader = UnitReader::<P>::new(reader);

        let mut accumulator = I::null_accumulator(inst);

        for i in 0..reader.length() {
            let item = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, ReduceStep::Identity);
        }

        accumulator
    }
}
