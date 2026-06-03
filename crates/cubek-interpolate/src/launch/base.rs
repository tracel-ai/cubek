use crate::{
    InterpolateError,
    {
        components::global::{execute_interpolate, execute_interpolate_nearest_backward},
        definition::{
            InterpolateForwardProblem, InterpolateOptions, NearestMode, accumulator_dtype,
        },
        launch::InterpolateStrategy,
        routines::{
            ForwardRoutine, GlobalMemoryRoutine, InterpolateBlueprint, SharedMemoryRoutine,
        },
    },
};
use cubecl::{
    calculate_cube_count_elemwise,
    ir::{ElemType, StorageType, UIntKind},
    prelude::*,
    std::FastDivmod,
    std::tensor::layout::linear::{LinearLayoutLaunch, LinearViewLayoutLaunch},
    tensor_vector_size_parallel,
};

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

    let problem = InterpolateForwardProblem::from_input_output_shapes(
        &input.shape,
        &[output.shape[1], output.shape[2]],
        options,
    );

    assert!(
        vector_size <= problem.channels,
        "Vector size {} is too large for the number of channels {}",
        vector_size,
        problem.channels
    );

    let address_type = input
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    let (blueprint, settings) = match strategy {
        InterpolateStrategy::GlobalMemoryStrategy(strategy) => GlobalMemoryRoutine::prepare(
            client,
            &problem,
            strategy,
            bytes_per_element,
            vector_size,
        )?,
        InterpolateStrategy::SharedMemoryStrategy(strategy) => SharedMemoryRoutine::prepare(
            client,
            &problem,
            strategy,
            bytes_per_element,
            vector_size,
        )?,
    };

    let num_vectors = settings.num_vectors;
    let cubes_per_batch = settings.num_tiles_width * settings.num_tiles_height;

    unsafe {
        interpolate_kernel::launch_unchecked(
            client,
            settings.cube_count,
            settings.cube_dim,
            address_type,
            vector_size,
            input.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            num_vectors,
            cubes_per_batch,
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
    num_vectors: FastDivmod<usize>,
    cubes_per_batch: FastDivmod<usize>,
    #[comptime] blueprint: InterpolateBlueprint,
    #[define(EI)] _dtype: StorageType,
    #[define(EA)] _acc_dtype: StorageType,
) {
    execute_interpolate::<(EI, EA), N>(input, output, num_vectors, cubes_per_batch, blueprint);
}

pub fn interpolate_nearest_backward_launch<R: Runtime>(
    client: &ComputeClient<R>,
    out_grad: TensorBinding<R>,
    output: TensorBinding<R>,
    nearest_mode: NearestMode,
    dtype: StorageType,
) -> Result<(), InterpolateError> {
    let vector_size = tensor_vector_size_parallel(
        client.io_optimized_vector_sizes(dtype.size()),
        &out_grad.shape,
        &out_grad.strides,
        out_grad.shape.len() - 1,
    );
    let shape_out = shape_divmod(&output);
    let out_layout = linear_layout(&output, vector_size);

    let working_units = output.shape.iter().product::<usize>() / vector_size as usize;
    let cube_dim = CubeDim::new(client, working_units);
    let cube_count = calculate_cube_count_elemwise(client, working_units, cube_dim);

    let address_type = out_grad
        .required_address_type(dtype.size())
        .max(output.required_address_type(dtype.size()));

    unsafe {
        execute_interpolate_nearest_backward::launch_unchecked(
            client,
            cube_count,
            cube_dim,
            address_type,
            vector_size,
            out_grad.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            shape_out,
            out_layout,
            nearest_mode,
            dtype,
        )
    };

    Ok(())
}

fn shape_divmod<R: Runtime>(binding: &TensorBinding<R>) -> SequenceArg<R, FastDivmod<usize>> {
    let mut out_seq = SequenceArg::new();
    for dim in binding.shape.iter() {
        out_seq.push(*dim);
    }
    out_seq
}

fn linear_layout<R: Runtime>(
    binding: &TensorBinding<R>,
    vector_size: usize,
) -> LinearLayoutLaunch<R> {
    LinearLayoutLaunch::from_shape_strides(
        binding.shape.clone(),
        binding.strides.clone(),
        // Don't care about type size, only vector size.
        Type::new(StorageType::Scalar(ElemType::UInt(UIntKind::U32))).with_vector_size(vector_size),
        LinearViewLayoutLaunch::new(),
    )
}
