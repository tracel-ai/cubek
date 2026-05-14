use crate::{
    InterpolateError,
    components::global::{TileSize, interpolate_kernel},
    definition::{InterpolateMode, InterpolateOptions},
};
use cubecl::{prelude::*, tensor_vector_size_parallel};

pub(crate) fn interpolate_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    let batch_size = output.shape[0];
    let out_h = output.shape[1];
    let out_w = output.shape[2];
    let channels = output.shape[3];

    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );

    // Using 16x16 tiles as a starting point
    let tile_w = 16.min(out_w);
    let tile_h = 16.min(out_h);

    let cube_dim = CubeDim::new_2d(tile_w as u32, tile_h as u32);

    let cubes_x = out_w.div_ceil(tile_w);
    let cubes_y = out_h.div_ceil(tile_h);

    let cubes_z = batch_size * channels / vector_size as usize;
    let cube_count = CubeCount::Static(cubes_x as u32, cubes_y as u32, cubes_z as u32);

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    let out_tile_size = TileSize::new(tile_h, tile_w);
    let in_tile_size = out_tile_size.to_input_tile(
        get_ratio(input.shape[1], output.shape[1]),
        get_ratio(input.shape[2], output.shape[2]),
    );

    interpolate_kernel::launch(
        client,
        cube_count,
        cube_dim,
        address_type,
        vector_size,
        input.into_tensor_arg(),
        output.into_tensor_arg(),
        options,
        out_tile_size,
        in_tile_size,
        dtype,
    );

    Ok(())
}

fn get_ratio(in_size: usize, out_size: usize) -> f32 {
    in_size as f32 / out_size as f32
}
