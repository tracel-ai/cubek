use crate::{
    InterpolateError,
    {
        components::global::execute_interpolate,
        definition::{
            InterpolateForwardProblem, InterpolateOptions, InterpolateProblem, accumulator_dtype,
        },
        launch::{InterpolateStrategy, RoutineStrategy},
        routines::{GlobalMemoryRoutine, InterpolateBlueprint, Routine, SharedMemoryRoutine},
    },
};
use cubecl::{prelude::*, std::FastDivmod, tensor_vector_size_parallel};

pub fn interpolate_launch<R: Runtime>(
    client: &ComputeClient<R>,
    input: TensorBinding<R>,
    output: TensorBinding<R>,
    options: InterpolateOptions,
    strategy: InterpolateStrategy,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    let acc_dtype = accumulator_dtype(dtype);
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &input.shape,
        &input.strides,
        input.shape.len() - 1,
    );
    let bytes_per_element = acc_dtype.size() * vector_size as usize;

    let problem = InterpolateProblem::Forward(InterpolateForwardProblem {
        input_shape: [
            input.shape[0],
            input.shape[1],
            input.shape[2],
            input.shape[3],
        ],
        output_size: [output.shape[2], output.shape[1]],
        options,
    });

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    let (blueprint, settings) = match strategy.routine {
        RoutineStrategy::GlobalMemoryStrategy(strategy) => {
            let routine = GlobalMemoryRoutine;
            routine.prepare(client, problem, strategy, bytes_per_element, vector_size)?
        }
        RoutineStrategy::SharedMemoryStrategy(strategy) => {
            let routine = SharedMemoryRoutine;
            routine.prepare(client, problem, strategy, bytes_per_element, vector_size)?
        }
    };

    let (output_width, output_height) = (output.shape[2], output.shape[1]);
    let channel_groups = output.shape[3] / vector_size;
    let num_tiles_x = output_width.div_ceil(settings.tile_size.width());
    let num_tiles_y = output_height.div_ceil(settings.tile_size.height());
    let cube_shape = get_cube_shape(
        channel_groups,
        settings.tile_size.area(),
        num_tiles_x * num_tiles_y,
    );

    println!(
        "Launching with strategy: {:?}, settings: {:?}",
        strategy, settings
    );

    unsafe {
        interpolate_kernel::launch_unchecked(
            client,
            settings.cube_count,
            settings.cube_dim,
            address_type,
            vector_size,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            cube_shape,
            blueprint,
            dtype,
            acc_dtype,
        )
    };

    Ok(())
}

#[cube(launch_unchecked, address_type = "dynamic")]
fn interpolate_kernel<EI: Float, EA: Float, N: Size>(
    input: &Tensor<Vector<EI, N>>,
    output: &mut Tensor<Vector<EI, N>>,
    cube_shape: Sequence<FastDivmod<usize>>,
    #[comptime] blueprint: InterpolateBlueprint,
    #[define(EI)] _dtype: StorageType,
    #[define(EA)] _acc_dtype: StorageType,
) {
    execute_interpolate::<(EI, EA), N>(input, output, cube_shape, blueprint);
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
