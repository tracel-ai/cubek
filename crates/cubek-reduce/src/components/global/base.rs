use crate::{IdleMode, LineMode, ReducePrecision};
use cubecl::{
    prelude::*,
    std::{CubeOption, tensor::r#virtual::VirtualTensor},
};

#[cube]
pub(crate) fn reduce_count(
    output_size: usize,
    #[comptime] line_mode: LineMode,
    #[comptime] input_line_size: LineSize,
) -> usize {
    match line_mode {
        LineMode::Parallel => output_size,
        LineMode::Perpendicular => output_size / input_line_size,
    }
}

#[cube]
pub fn idle_check<P: ReducePrecision, Out: Numeric>(
    input: &VirtualTensor<P::EI>,
    output: &mut VirtualTensor<Out, ReadWrite>,
    reduce_index_start: usize,
    #[comptime] line_mode: LineMode,
    #[comptime] idle_mode: IdleMode,
) -> CubeOption<bool> {
    if idle_mode.is_enabled() {
        let reduce_count = reduce_count(
            output.len() * output.line_size(),
            line_mode,
            input.line_size(),
        );

        match idle_mode {
            IdleMode::None => CubeOption::new_None(),
            IdleMode::Mask => CubeOption::new_Some(reduce_index_start >= reduce_count),
            IdleMode::Terminate => {
                if reduce_index_start >= reduce_count {
                    terminate!();
                }
                CubeOption::new_None()
            }
        }
    } else {
        CubeOption::new_None()
    }
}
