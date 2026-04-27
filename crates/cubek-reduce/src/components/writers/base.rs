use crate::{
    ReduceInstruction, ReducePrecision, VectorizationMode,
    components::{
        args::NumericVector,
        instructions::{Accumulator, AccumulatorFormat},
        writers::{parallel::ParallelWriter, perpendicular::PerpendicularWriter},
    },
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
/// Abstract how data is written to global memory.
///
/// Depending on the problem kind, writes might be buffered to optimize vectorization, only
/// happening when [Writer::commit()] is called.
pub enum Writer<Out: NumericVector> {
    Parallel(ParallelWriter<Out>),
    Perpendicular(PerpendicularWriter<Out>),
}

#[cube]
impl<Out: NumericVector> Writer<Out> {
    pub fn new<P: ReducePrecision>(
        input: &VirtualTensor<P::EI, P::SI>,
        output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
        reduce_axis: usize,
        out_vec_axis: usize,
        write_index: usize,
        #[comptime] vectorization_mode: VectorizationMode,
        #[comptime] acc_format: AccumulatorFormat,
    ) -> Writer<Out> {
        match vectorization_mode {
            VectorizationMode::Parallel => {
                Writer::<Out>::new_Parallel(ParallelWriter::<Out>::new::<P>(
                    input,
                    output,
                    reduce_axis,
                    out_vec_axis,
                    write_index,
                    acc_format,
                ))
            }
            VectorizationMode::Perpendicular => {
                Writer::<Out>::new_Perpendicular(PerpendicularWriter::<Out>::new::<P>(
                    input,
                    output,
                    reduce_axis,
                    out_vec_axis,
                    write_index,
                    acc_format,
                ))
            }
        }
    }

    pub fn write<P: ReducePrecision, I: ReduceInstruction<P>>(
        &mut self,
        local_index: usize,
        accumulator: Accumulator<P>,
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
