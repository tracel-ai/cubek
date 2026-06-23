use crate::{
    components::resample_kernel,
    definition::{Resample, ResampleArgsLaunch},
};
use cubecl::{
    prelude::*,
    server::CubeCountSelection,
    std::{
        FastDivmod,
        tensor::{
            launch::ViewArg,
            layout::{
                CoordsDynI,
                fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
            },
        },
    },
    tensor_vector_size_parallel,
    zspace::Shape,
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

    let tile_shape = compute_tile_shape(&output.shape, &cube_dim, vectorized_axis, vector_size);

    let tile_strides = compute_strides(&tile_shape);

    let cube_shape = compute_cube_shape(&output.shape, &tile_shape, vectorized_axis, vector_size);

    let cube_strides = compute_strides(&cube_shape);

    let cube_count = calculate_cube_count(client, &cube_shape);

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            vector_size,
            view(input, vector_size),
            view(output, vector_size),
            to_sequence::<R, FastDivmod<usize>>(&tile_shape),
            to_sequence::<R, FastDivmod<usize>>(&tile_strides),
            to_sequence::<R, FastDivmod<usize>>(&cube_shape),
            to_sequence::<R, FastDivmod<usize>>(&cube_strides),
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
    let rank = input.shape.len();

    for axis in (0..rank).rev() {
        let in_vec = tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(dtype.size()),
            &input.shape,
            &input.strides,
            axis,
        );

        let out_vec = tensor_vector_size_parallel(
            client.io_optimized_vector_sizes(dtype.size()),
            &output.shape,
            &output.strides,
            axis,
        );

        let vector_size = in_vec.min(out_vec);

        if vector_size > 1 {
            return (vector_size, axis);
        }
    }

    // Fallback if no axis can be vectorized.
    (1, rank.saturating_sub(1))
}

/// Computes the tile shape for the given output shape, cube dimension, vectorized axis, and vector size.
fn compute_tile_shape(
    output_shape: &Shape,
    cube_dim: &CubeDim,
    vectorized_axis: usize,
    vector_size: usize,
) -> Shape {
    let len = output_shape.len();

    let mut tile_shape = Shape::from(vec![1; len]);

    let mut remaining_cube_dim = cube_dim.num_elems() as usize;

    // Process dimensions in reverse order to ensure a cube processes contiguous memory (memory coalescing).
    for i in (0..len).rev() {
        let size = if vectorized_axis == i {
            output_shape[i] / vector_size
        } else {
            output_shape[i]
        };

        // This strategy ensure that the product of tile_shapes >= the original cube_dim.
        // Which guarantee that each thread will have at least one element to process.
        tile_shape[i] = size.min(remaining_cube_dim).max(1);

        remaining_cube_dim = remaining_cube_dim.div_ceil(tile_shape[i]);
    }

    tile_shape
}

/// Computes the cube shape for the given output shape, tile shape, vectorized axis, and vector size.
fn compute_cube_shape(
    output_shape: &Shape,
    tile_shape: &Shape,
    vectorized_axis: usize,
    vector_size: usize,
) -> Shape {
    let len = output_shape.len();

    let mut cube_shape = Shape::from(vec![1; len]);

    for i in (0..len).rev() {
        let size = if vectorized_axis == i {
            output_shape[i] / vector_size
        } else {
            output_shape[i]
        };

        cube_shape[i] = size.div_ceil(tile_shape[i]);
    }

    cube_shape
}

/// Helper to compute row-major stride from a shape.
fn compute_strides(shape: &Shape) -> Shape {
    let len = shape.len();

    let mut strides = Shape::from(vec![1; len]);

    if len == 0 {
        return strides;
    }

    // Iterate backwards starting from the second-to-last element
    for i in (0..len - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }

    strides
}

/// Calculate the number of cubes required to execute an operation where one cube unit is
/// assigned to one tile.
fn calculate_cube_count<R: Runtime>(client: &ComputeClient<R>, cube_shape: &Shape) -> CubeCount {
    if cube_shape.is_empty() {
        return CubeCount::Static(0, 0, 0);
    }
    CubeCountSelection::new(client, cube_shape.num_elements() as u32).cube_count()
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
fn view<R: Runtime>(tensor: TensorBinding<R>, vector_size: VectorSize) -> ViewArg<CoordsDynI, R> {
    let shape_seq = tensor
        .shape
        .iter()
        .map(|&s| s as i32)
        .collect::<SequenceArg<R, i32>>();

    let layout = FixedDimLayoutLaunch::<CoordsDynI, R>::from_shape_handle_unchecked(
        &tensor,
        shape_seq,
        vector_size,
    );
    let buffer = tensor.into_tensor_arg();
    ViewArg::new_tensor::<FixedDimLayout<CoordsDynI>>(buffer, layout)
}
