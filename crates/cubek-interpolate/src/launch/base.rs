use crate::{
    InterpolateError,
    {
        components::global::{TileSize, interpolate_kernel},
        definition::InterpolateOptions,
    },
};
use cubecl::{prelude::*, std::FastDivmod, tensor_vector_size_parallel};

pub(crate) fn interpolate_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);

    let output_tile_size = TileSize::new(cube_dim.x as usize, cube_dim.y as usize, options);

    let batch = output.shape[0];
    let height = output.shape[1];
    let width = output.shape[2];
    let channel_groups = output.shape[3] / vector_size;

    let num_tiles_x = width.div_ceil(cube_dim.x as usize);
    let num_tiles_y = height.div_ceil(cube_dim.y as usize);

    let cube_count = CubeCount::Static(
        (num_tiles_x * channel_groups) as u32,
        num_tiles_y as u32,
        batch as u32,
    );

    let threads_per_cube = output_tile_size.area();
    let cubes_per_batch = num_tiles_x * num_tiles_y;
    let cube_shape = get_cube_shape(channel_groups, threads_per_cube, cubes_per_batch);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        interpolate_kernel::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            cube_shape,
            output_tile_size,
            options,
            dtype,
        )
    };

    Ok(())
}

fn get_cube_shape<R: Runtime>(
    channel_groups: usize,
    threads_per_cube: usize,
    cubes_per_batch: usize,
) -> SequenceArg<R, FastDivmod<usize>> {
    let mut cube_shape = SequenceArg::new();
    cube_shape.push(channel_groups);
    cube_shape.push(threads_per_cube);
    cube_shape.push(cubes_per_batch);
    cube_shape
}
