use crate::{
    components::resample_kernel,
    definition::{Resample, ResampleArgsLaunch, TileSizeLauncher},
};
use cubecl::{
    prelude::*,
    server::CubeCountSelection,
    std::tensor::{
        launch::ViewArg,
        layout::{
            CoordsDynI,
            fixed_dim::{FixedDimLayout, FixedDimLayoutLaunch},
        },
    },
    tensor_vector_size_parallel,
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

    let (tile_size, cube_size) =
        TileSizeLauncher::new(&output.shape, &cube_dim, vectorized_axis, vector_size);

    let cube_count = calculate_cube_count(client, &cube_size);

    unsafe {
        resample_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            vector_size,
            view(input, vector_size),
            view(output, vector_size),
            tile_size.to_launch(),
            cube_size.to_launch(),
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

/// Calculate the number of cubes required to execute an operation where one cube unit is
/// assigned to one tile.
fn calculate_cube_count<R: Runtime>(
    client: &ComputeClient<R>,
    cube_size_launcher: &TileSizeLauncher,
) -> CubeCount {
    if cube_size_launcher.is_empty() {
        return CubeCount::Static(0, 0, 0);
    }
    CubeCountSelection::new(client, cube_size_launcher.num_cubes() as u32).cube_count()
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
