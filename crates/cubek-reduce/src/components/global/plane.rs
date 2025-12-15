use crate::{
    LineMode, ReduceInstruction, ReducePrecision,
    components::{
        global::reduce_count,
        instructions::reduce_inplace,
        readers::{Reader, plane::PlaneReader},
        writer,
    },
    routines::PlaneReduceBlueprint,
};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[derive(CubeType)]
pub struct GlobalFullPlaneReduce;

#[cube]
impl GlobalFullPlaneReduce {
    pub fn execute<P: ReducePrecision, Out: Numeric, I: ReduceInstruction<P>>(
        input: &VirtualTensor<P::EI>,
        output: &mut VirtualTensor<Out, ReadWrite>,
        reduce_axis: u32,
        inst: &I,
        #[comptime] line_mode: LineMode,
        #[comptime] blueprint: PlaneReduceBlueprint,
    ) {
        let reduce_index = CUBE_POS * CUBE_DIM_Y + UNIT_POS_Y;

        if comptime![blueprint.plane_idle] {
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
            blueprint.bound_checks,
            line_mode,
        );
        let reader = PlaneReader::<P>::new(reader);

        let mut accumulator = I::null_accumulator(inst, input_line_size);

        for i in 0..reader.length() {
            let (item, coordinate) = reader.read(i);
            reduce_inplace::<P, I>(
                inst,
                &mut accumulator,
                item,
                coordinate,
                comptime!(!blueprint.independent),
            );
        }

        let result = match blueprint.independent {
            true => {
                let (item, coordinate) = I::read_accumulator(inst, &accumulator);
                let mut result = I::null_accumulator(inst, input_line_size);
                reduce_inplace::<P, I>(inst, &mut result, item, coordinate, true);
                result
            }
            false => accumulator,
        };

        if UNIT_POS_X == 0 {
            writer::write_accumulator::<P, Out, I>(
                output,
                result,
                reduce_index,
                input.shape(reduce_axis),
                line_mode,
                input.line_size(),
                inst,
            )
        }
    }
}
