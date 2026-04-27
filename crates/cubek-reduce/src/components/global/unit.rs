use crate::{
    BoundChecks, ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        global::{idle_check, reduction_output_base},
        instructions::{Accumulator, ReduceStep, reduce_inplace},
        readers::{Reader, unit::UnitReader},
        writers::Writer,
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
            output,
            reduce_axis,
            comptime!(acc_format.len()),
        );

        let mut writer = Writer::<Out>::new::<P>(
            input,
            output,
            reduce_axis,
            out_vec_axis,
            write_index,
            vectorization_mode,
            acc_format,
        );

        let write_count = writer.write_count();
        let reduce_index_start = write_index * write_count;

        let idle = idle_check::<P, Out>(
            input,
            output,
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
            writer.write::<P, I>(b, accumulator, inst);
        }

        writer.commit();
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
