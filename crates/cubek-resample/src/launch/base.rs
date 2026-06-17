use crate::{
    components::resample_kernel,
    definition::{Resample, ResampleArgsLaunch, TileArgsLaunch},
};
use cubecl::{
    calculate_cube_count_elemwise,
    prelude::*,
    std::tensor::{
        launch::ViewArg,
        layout::{
            CoordsDyn,
            fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
        },
    },
};

/// Launch the resample kernel for a single spatial axis.
pub fn resample_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    args: ResampleArgsLaunch<R>,
    config: Resample,
    dtype: StorageType,
) {
    let (vector_size, vectorized_axis) = vectorize(client, &input, &output, dtype);

    let working_units = output.shape.iter().product::<usize>() / vector_size;

    let cube_dim = CubeDim::new(client, working_units);

    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let tile_args = compute_tile_args(&output, &cube_dim, vectorized_axis, vector_size);

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            vector_size,
            view(input, vector_size),
            view(output, vector_size),
            tile_args,
            args,
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

/// Distributes the workload between threads in a tiled layout.
fn compute_tile_args<R: Runtime>(
    output: &TensorBinding<R>,
    cube_dim: &CubeDim,
    vectorized_axis: usize,
    vector_size: usize,
) -> TileArgsLaunch<R> {
    let len = output.shape.len();

    let mut tile_size = vec![1; len];
    let mut cube_size = vec![1; len];

    let mut remaining_cube_dim = cube_dim.num_elems() as usize;

    // Process dimensions in reverse order to ensure a cube processes contiguous memory (memory coalescing).
    for i in (0..len).rev() {
        let size = if vectorized_axis == i {
            output.shape[i] / vector_size
        } else {
            output.shape[i]
        };

        // This strategy ensure that the product of tile_sizes >= the original cube_dim.
        // Which guarantee that each thread will have at least one element to process.
        tile_size[i] = size.min(remaining_cube_dim).max(1);
        cube_size[i] = size.div_ceil(tile_size[i]);

        remaining_cube_dim = remaining_cube_dim.div_ceil(tile_size[i]);
    }

    let tile_strides = compute_strides(&tile_size);
    let cube_strides = compute_strides(&cube_size);

    TileArgsLaunch::new(
        to_sequence(&tile_size),
        to_sequence(&tile_strides),
        to_sequence(&cube_size),
        to_sequence(&cube_strides),
        to_sequence(&output.shape),
    )
}

/// Helper to compute row-major stride from a shape.
fn compute_strides(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];

    // Iterate backwards starting from the second-to-last element
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

/// Convert a slice of dimensions into a `SequenceArg`.
fn to_sequence<R: Runtime, T: LaunchArg>(shape: &[usize]) -> SequenceArg<R, T>
where
    usize: Into<<T as LaunchArg>::RuntimeArg<R>>,
{
    let mut sequence = SequenceArg::new();
    for dim in shape.iter() {
        sequence.push((*dim).into());
    }
    sequence
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
