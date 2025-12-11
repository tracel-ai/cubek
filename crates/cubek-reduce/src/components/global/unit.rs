use crate::{
    BoundChecksInner, LineMode, ReduceInstruction, ReducePrecision,
    components::{
        instructions::reduce_inplace,
        readers::{Reader, unit::UnitReader},
        writer,
    },
    routines::ReduceBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullUnitReduce;

#[cube]
impl GlobalFullUnitReduce {
    pub fn execute<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        reduce_index: u32,
        inst: &I,
        #[comptime] blueprint: ReduceBlueprint,
    ) {
        #[allow(clippy::collapsible_if)]
        if comptime![blueprint.bound_checks] {
            if reduce_index
                >= get_reduce_count(
                    output.len() * output.line_size(),
                    blueprint.line_mode,
                    input.line_size(),
                )
            {
                terminate!();
            }
        }
        let input_line_size = input.line_size();

        let reader = Reader::<P>::new::<I, Out>(
            input,
            output,
            inst,
            reduce_axis,
            reduce_index,
            comptime!(BoundChecksInner::None),
            blueprint.line_mode,
        );
        let reader = UnitReader::<P>::new(reader);
        let mut accumulator = I::null_accumulator(inst, input_line_size);

        for i in 0..reader.len() {
            let (item, coordinate) = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, coordinate, false);
        }

        writer::write::<P, Out, I>(
            output,
            accumulator,
            reduce_index,
            input.shape(reduce_axis),
            blueprint,
            input.line_size(),
            inst,
        )
    }
}

#[cube]
fn get_reduce_count(
    output_size: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] input_line_size: u32,
) -> u32 {
    match comptime!(line_mode) {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / input_line_size,
    }
}
