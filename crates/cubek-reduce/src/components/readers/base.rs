use crate::{
    BoundChecksInner, LineMode, ReduceInstruction, ReducePrecision,
    components::readers::{parallel::ParallelReader, perpendicular::PerpendicularReader},
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub enum Reader<P: ReducePrecision> {
    Parallel(ParallelReader<P>),
    Perpendicular(PerpendicularReader<P>),
}

#[cube]
impl<P: ReducePrecision> Reader<P> {
    pub fn new<I: ReduceInstruction<P>, Out: Numeric>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        inst: &I,
        reduce_axis: u32,
        reduce_index: u32,
        #[comptime] bound_checks: BoundChecksInner,
        #[comptime] line_mode: LineMode,
    ) -> Reader<P> {
        match line_mode {
            LineMode::Parallel => Reader::<P>::new_Parallel(ParallelReader::<P>::new::<I, Out>(
                input,
                output,
                inst,
                reduce_axis,
                reduce_index,
                bound_checks,
            )),
            LineMode::Perpendicular => {
                Reader::<P>::new_Perpendicular(PerpendicularReader::<P>::new::<I, Out>(
                    input,
                    output,
                    inst,
                    reduce_axis,
                    reduce_index,
                    bound_checks,
                ))
            }
        }
    }
}
