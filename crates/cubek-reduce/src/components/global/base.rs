use crate::{IdleMode, ReducePrecision, VectorizationMode, components::args::NumericVector};
use cubecl::{prelude::*, std::tensor::r#virtual::VirtualTensor};

/// Output offset (in vector units, matching the writer's layout) of the first
/// slot (k=0) of reduction `reduction_index`.
#[cube]
pub fn reduction_output_base<T: Numeric, N: Size>(
    reduction_index: usize,
    output: &mut VirtualTensor<T, N, ReadWrite>,
    reduce_axis: usize,
    #[comptime] accumulator_length: usize,
) -> usize {
    if comptime![accumulator_length > 1] {
        let slot_stride = output.stride(reduce_axis) / output.vector_size();
        let group_stride = slot_stride * accumulator_length;
        (reduction_index / slot_stride) * group_stride + (reduction_index % slot_stride)
    } else {
        reduction_index
    }
}

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
pub fn idle_check<P: ReducePrecision, Out: NumericVector>(
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
