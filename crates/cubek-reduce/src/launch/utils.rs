use crate::{LineMode, launch::LineSizeStrategy};
use cubecl::{
    prelude::*, std::tensor::is_contiguous, tensor_line_size_parallel,
    tensor_line_size_perpendicular,
};

/// Calculate the number of planes in a cube.
pub fn calculate_plane_count_per_cube(
    working_units: usize,
    plane_dim: u32,
    num_cpu_cores: Option<u32>,
) -> u32 {
    match num_cpu_cores {
        Some(num_cores) => core::cmp::min(num_cores, working_units as u32),
        None => {
            let plane_count_max = core::cmp::max(1, working_units / plane_dim as usize);

            // Ensures `plane_count` is a power of 2.
            const NUM_PLANE_MAX: u32 = 8u32;
            const NUM_PLANE_MAX_LOG2: u32 = NUM_PLANE_MAX.ilog2();
            let plane_count_max_log2 =
                core::cmp::min(NUM_PLANE_MAX_LOG2, usize::ilog2(plane_count_max));
            2u32.pow(plane_count_max_log2)
        }
    }
}

pub fn generate_line_size<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorHandleRef<R>,
    output: &TensorHandleRef<R>,
    axis: usize,
    dtype: StorageType,
    line_mode: LineMode,
    strategy: &LineSizeStrategy,
) -> (usize, usize) {
    let line_size_input = match line_mode {
        LineMode::Parallel => tensor_line_size_parallel(
            client.io_optimized_line_sizes_unchecked(dtype.size()),
            input.shape,
            input.strides,
            axis,
        ),
        LineMode::Perpendicular => {
            // To compute the maximum line size we can used,
            // we first sort both the input and output axes by increasing strides.
            // As example, consider
            //    input shape = [2, 4, 6, 8]
            //    input stride = [1, 16, 64, 2]
            //    output shape = [2, 1, 6, 8]
            //    output stride = [1, 1, 2, 12]
            //    axis = 1
            //
            // then we have
            //    input sorted axis = [0, 3, 1, 2]
            //    output sorted axis = [0, 1, 2, 3]
            //
            // From that point, we look at all the axes before the target axis in the sorted input.
            // That is [0, 3] in the example.
            // In the output, we remove the target axis leading to [0, 2, 3] in the example.
            //
            // In order to use perpendicular line, we are limited by the number of entries that are both
            // contiguous in the input and output. This is obtained by taking the head of each list until they are different.
            // In the above example, only the 0 axis is contiguous in both tensor, but it output sorted axis were [0, 1, 3, 2] instead,
            // both the 0 and 3 axes would be contiguous in the two tensors.
            // The corresponding number of entries is the product of the shape for the contiguous axes.
            // In the example, it is simply 2.
            //
            // This gives us an upper bound on the line size we can used.
            // Then, we use the regular method to find the best line size that match the device capacities.

            let mut input_axis_and_strides = input.strides.iter().enumerate().collect::<Vec<_>>();
            input_axis_and_strides.sort_by_key(|(_, stride)| *stride);
            let input_sorted_axis = input_axis_and_strides
                .into_iter()
                .map(|(a, _)| a)
                .take_while(|a| *a != axis);

            let mut output_axis_and_strides = output.strides.iter().enumerate().collect::<Vec<_>>();
            output_axis_and_strides.sort_by_key(|(_, stride)| *stride);
            let output_sorted_axis = output_axis_and_strides
                .into_iter()
                .filter_map(|(a, _)| (a != axis).then_some(a));

            let max_line_size = input_sorted_axis
                .zip(output_sorted_axis)
                .filter_map(|(i, o)| (i == o).then_some(output.shape[i]))
                .product();

            match client.properties().hardware.num_cpu_cores.is_some() {
                true => {
                    // On CPU we benefit from bigger line size, which increases the number of
                    // consecutive loads from global memory on perpendicular reduce.
                    let supported_line_sizes = R::supported_line_sizes()
                        .iter()
                        .filter(|size| **size <= max_line_size && max_line_size % **size == 0)
                        .copied();

                    tensor_line_size_perpendicular(
                        supported_line_sizes,
                        input.shape,
                        input.strides,
                        axis,
                    )
                }
                false => {
                    let supported_line_sizes = client
                        .io_optimized_line_sizes_unchecked(dtype.size())
                        .filter(|&size| {
                            size <= max_line_size && max_line_size.is_multiple_of(size)
                        });

                    tensor_line_size_perpendicular(
                        supported_line_sizes,
                        input.shape,
                        input.strides,
                        axis,
                    )
                }
            }
        }
    };

    let mut line_size_output = 1;

    if line_size_input > 1 && line_mode == LineMode::Perpendicular {
        // TODO that this can be improved
        let rank = output.strides.len();
        let is_contiguous = is_contiguous(&output.shape[axis..rank], &output.strides[axis..rank])
            && output.strides[rank - 1] == 1;
        let shape = output.shape.get(axis + 1).copied().unwrap_or(1);

        if is_contiguous && shape.is_multiple_of(line_size_input) {
            line_size_output = line_size_input;
        }
    }

    if strategy.parallel_output_vectorization
        && line_mode == LineMode::Parallel
        && line_size_input > 1
        && is_contiguous(input.shape, input.strides)
        && axis == input.shape.len() - 1
    {
        let supported_line_sizes = client.io_optimized_line_sizes_unchecked(dtype.size());
        let num_reduce = output.shape.iter().copied().product::<usize>();
        line_size_output = supported_line_sizes
            .filter(|&line_size| num_reduce % line_size == 0)
            .max()
            .unwrap_or(1);
    }

    (line_size_input, line_size_output)
}
