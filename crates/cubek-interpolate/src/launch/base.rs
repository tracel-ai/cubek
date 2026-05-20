use crate::{
    InterpolateError,
    {
        components::global::{TileSize, interpolate_kernel},
        definition::{InterpolateOptions, MemoryStrategy, accumulator_dtype, get_halo},
    },
};
use cubecl::{prelude::*, std::FastDivmod, tensor_vector_size_parallel};

pub(crate) fn interpolate_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    dtype: StorageType,
    memory_strategy: MemoryStrategy,
) -> Result<(), InterpolateError> {
    let acc_dtype = accumulator_dtype(dtype);
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
    let (output_width, output_height) = (output.shape[2], output.shape[1]);
    let (input_width, input_height) = (input.shape[2], input.shape[1]);
    let channel_groups = output.shape[3] / vector_size;

    let (smem_width, smem_height) = compute_smem_size(
        input_width,
        input_height,
        output_width,
        output_height,
        memory_strategy,
        options,
        output_tile_size,
    );

    let num_tiles_x = output_width.div_ceil(cube_dim.x as usize);
    let num_tiles_y = output_height.div_ceil(cube_dim.y as usize);

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
            memory_strategy,
            smem_width,
            smem_height,
            dtype,
            acc_dtype,
        )
    };

    Ok(())
}

fn compute_smem_size(
    input_width: usize,
    input_height: usize,
    output_width: usize,
    output_height: usize,
    memory_strategy: MemoryStrategy,
    options: InterpolateOptions,
    output_tile_size: TileSize,
) -> (usize, usize) {
    match memory_strategy {
        MemoryStrategy::Shared => {
            let halo = get_halo(options.mode);
            let radius_offset = (halo - 1) / 2;

            let tile_w = output_tile_size.width().max(1) as f64;
            let tile_h = output_tile_size.height().max(1) as f64;

            let scale_x = if options.align_corners {
                if output_width > 1 {
                    (input_width - 1) as f64 / (output_width - 1) as f64
                } else {
                    0.0
                }
            } else {
                input_width as f64 / output_width as f64
            };

            let scale_y = if options.align_corners {
                if output_height > 1 {
                    (input_height - 1) as f64 / (output_height - 1) as f64
                } else {
                    0.0
                }
            } else {
                input_height as f64 / output_height as f64
            };

            let span_x = (scale_x * (tile_w - 1.0)).max(0.0) + 1.0;
            let span_y = (scale_y * (tile_h - 1.0)).max(0.0) + 1.0;

            let smem_w = span_x.ceil() as usize + 2 * radius_offset;
            let smem_h = span_y.ceil() as usize + 2 * radius_offset;

            (smem_w.max(1), smem_h.max(1))
        }
        MemoryStrategy::Global => (0, 0),
    }
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
