use crate::{components::resample_kernel, definition::Resample};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::{
        FastDivmod,
        tensor::{
            launch::ViewArg,
            layout::{
                CoordsDyn,
                fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
            },
        },
    },
};

/// Launch the resample kernel for a single spatial axis.
pub fn resample_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    config: Resample,
    dtype: StorageType,
) {
    let (vector_size, vectorized_axis) = vectorize(client, &input, &output, dtype);

    let working_units = output.shape.iter().product::<usize>() / vector_size;

    let cube_dim = CubeDim::new(client, working_units);

    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let output_shape = divmod_sequence(&output.shape);
    let output_strides = divmod_sequence(&output.strides);

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            vector_size,
            view(input, vector_size),
            view(output, vector_size),
            output_shape,
            output_strides,
            working_units,
            config,
            vectorized_axis,
            dtype,
        );
    }
}

/// Returns the optimal vector size and the vectorized axis for the given tensors.
fn vectorize<R: Runtime>(
    client: &ComputeClient<R>,
    input: &TensorBinding<R>,
    output: &TensorBinding<R>,
    dtype: StorageType,
) -> (usize, usize) {
    let supported_sizes = client.io_optimized_vector_sizes(dtype.size());
    let rank = input.shape.len();

    for i in 1..=rank {
        let axis = rank - i;

        // Break and don't vectorize if the axis is not contiguous.
        if input.strides[axis] > 1 || output.strides[axis] > 1 {
            break;
        }

        // Find the largest vector size that works for both tensors on this axis
        for vector_size in supported_sizes.clone() {
            if vector_size == 1 {
                continue;
            }

            // If this vector size is supported by both, take it and break.
            if input.shape[axis].is_multiple_of(vector_size)
                && output.shape[axis].is_multiple_of(vector_size)
            {
                return (vector_size, axis);
            }
        }
    }

    // Fallback if no axis can be vectorized.
    (1, rank.saturating_sub(1))
}

/// Convert a sequence of shapes to a sequence of fast divmod.
fn divmod_sequence<R: Runtime>(shape: &[usize]) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

/// Convert a tensor binding to a view argument.
fn view<R: Runtime>(tensor: TensorBinding<R>, vector_size: VectorSize) -> ViewArg<CoordsDyn, R> {
    let shape_seq = tensor
        .shape
        .iter()
        .map(|&s| s as u32)
        .collect::<SequenceArg<R, u32>>();

    let layout = FixedDimLayoutLaunch::<CoordsDyn, R>::from_shape_handle_unchecked(
        &tensor,
        shape_seq,
        vector_size,
    );
    let buffer = tensor.into_tensor_arg();
    ViewArg::new_tensor::<FixedDimLayout<CoordsDyn>>(buffer, layout)
}
