use crate::{
    BoundChecks, LineMode, ReduceInstruction, ReducePrecision,
    components::{
        global::reduce_count,
        instructions::reduce_inplace,
        readers::{Reader, unit::UnitReader},
        writer,
    },
    routines::UnitReduceBlueprint,
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
        inst: &I,
        #[comptime] line_mode: LineMode,
        #[comptime] blueprint: UnitReduceBlueprint,
    ) {
        let reduce_index = CUBE_POS * CUBE_DIM + UNIT_POS;

        if comptime![blueprint.unit_idle] {
            let reduce_count = reduce_count(
                output.len() * output.line_size(),
                line_mode,
                input.line_size(),
            );

            if reduce_index >= reduce_count {
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
            comptime!(BoundChecks::None),
            line_mode,
        );
        let reader = UnitReader::<P>::new(reader);

        let mut accumulator = I::null_accumulator(inst, input_line_size);

        for i in 0..reader.length() {
            let (item, coordinate) = reader.read(i);
            reduce_inplace::<P, I>(inst, &mut accumulator, item, coordinate, false);
        }

        writer::write_accumulator::<P, Out, I>(
            output,
            accumulator,
            reduce_index,
            input.shape(reduce_axis),
            line_mode,
            input.line_size(),
            inst,
        )
    }
}
