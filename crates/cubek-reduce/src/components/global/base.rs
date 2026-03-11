use crate::{IdleMode, ReducePrecision, VectorizationMode, components::args::NumericLine};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

#[cube]
pub(crate) fn reduce_count(
    output_size: usize,
    #[comptime] vectorization_mode: VectorizationMode,
    #[comptime] input_vector_size: VectorSize,
) -> usize {
    match vectorization_mode {
        VectorizationMode::Parallel => output_size,
        VectorizationMode::Perpendicular => output_size / input_vector_size,
    }
}

#[cube]
pub fn idle_check<P: ReducePrecision, Out: NumericLine>(
    input: &VirtualTensor<P::EI, P::SI>,
    output: &mut VirtualTensor<Out::T, Out::N, ReadWrite>,
    reduce_index_start: usize,
    #[comptime] vectorization_mode: VectorizationMode,
    #[comptime] idle_mode: IdleMode,
) -> ComptimeOption<bool> {
    if idle_mode.is_enabled() {
        let reduce_count = reduce_count(
            output.len() * output.vector_size(),
            vectorization_mode,
            input.vector_size(),
        );

        match idle_mode {
            IdleMode::None => ComptimeOption::new_None(),
            IdleMode::Mask => ComptimeOption::new_Some(reduce_index_start >= reduce_count),
            IdleMode::Terminate => {
                if reduce_index_start >= reduce_count {
                    terminate!();
                }
                ComptimeOption::new_None()
            }
        }
    } else {
        ComptimeOption::new_None()
    }
}
