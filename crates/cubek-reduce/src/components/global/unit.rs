use crate::{
    BoundChecks, LineMode, ReduceInstruction, ReducePrecision,
    components::{
        global::reduce_count,
        instructions::reduce_inplace,
        readers::{Reader, unit::UnitReader},
        writer::Writer,
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
        let mut writer = Writer::<Out>::new(
            input.shape(reduce_axis),
            input.line_size(),
            output.line_size(),
            line_mode,
        );

        let num_accumulate = writer.num_accumulate();
        let reduce_index_start = ABSOLUTE_POS * num_accumulate;

        for b in 0..num_accumulate {
            let reduce_index = reduce_index_start + b;

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

            writer.accumulate::<P, I>(b, accumulator, inst);
        }

        writer.commit(output, ABSOLUTE_POS);
    }
}
