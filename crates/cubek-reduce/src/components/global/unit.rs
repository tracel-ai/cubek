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
        let write_index = ABSOLUTE_POS;
        let mut writer =
            Writer::<Out>::new::<P>(input, output, reduce_axis, write_index, line_mode);

        let write_count = writer.write_count();
        let reduce_index_start = write_index * write_count;

        if comptime![blueprint.unit_idle] {
            let reduce_count = reduce_count(
                output.len() * output.line_size(),
                line_mode,
                input.line_size(),
            );

            if reduce_index_start >= reduce_count {
                terminate!();
            }
        }

        for b in 0..write_count {
            let reduce_index = reduce_index_start + b;
            let accumulator = Self::reduce_single::<P, Out, I>(
                input,
                output,
                reduce_axis,
                reduce_index,
                inst,
                line_mode,
            );
            writer.write::<P, I>(b, accumulator, inst);
        }

        writer.commit();
    }

    fn reduce_single<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        reduce_index: u32,
        inst: &I,
        #[comptime] line_mode: LineMode,
    ) -> I::AccumulatorItem {
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

        accumulator
    }
}
