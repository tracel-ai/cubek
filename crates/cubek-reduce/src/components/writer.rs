use crate::{LineMode, ReduceInstruction, ReducePrecision};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub fn write_accumulator<P: ReducePrecision, Out: Numeric, R: ReduceInstruction<P>>(
    output: &mut VirtualTensor<Out, ReadWrite>,
    accumulator: R::AccumulatorItem,
    reduce_index: u32,
    shape_axis_reduce: u32,
    #[comptime] line_mode: LineMode,
    #[comptime] input_line_size: u32,
    inst: &R,
) {
    match comptime!(line_mode) {
        LineMode::Parallel => {
            let result = R::merge_line::<Out>(inst, accumulator, shape_axis_reduce);
            output.write(reduce_index, Line::cast_from(result))
        }
        LineMode::Perpendicular => {
            let out = R::to_output_perpendicular(inst, accumulator, shape_axis_reduce);
            let output_line_size = output.line_size();

            if comptime![output_line_size == input_line_size] {
                output.write(reduce_index, out);
            } else {
                let num_iters = comptime![input_line_size / output_line_size];

                #[unroll]
                for i in 0..num_iters {
                    let mut tmp = Line::empty(output_line_size);

                    #[unroll]
                    for j in 0..output_line_size {
                        tmp[j] = out[i * output_line_size + j];
                    }

                    let index = num_iters * reduce_index + i;
                    output.write(index, tmp);
                }
            }
        }
    }
}
