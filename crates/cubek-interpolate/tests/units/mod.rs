mod nearest;

use cubecl::{TestRuntime, client::ComputeClient, ir::StorageType, std::tensor::TensorHandle};
use cubek_interpolate::cpu_reference::cpu_reference_from_host;
use cubek_interpolate::definition::{InterpolateOptions, InterpolateProblem};
use cubek_interpolate::{InterpolateError, interpolate};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, assert_equals_approx,
};

pub fn build_output_shape(input: &HostData, output_size: [usize; 2]) -> Vec<usize> {
    vec![
        input.shape[0],
        output_size[0],
        output_size[1],
        input.shape[3],
    ]
}

pub fn build_output_tensor(
    client: &ComputeClient<TestRuntime>,
    output_shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), output_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

pub fn output_host_f32(
    client: &ComputeClient<TestRuntime>,
    output: TensorHandle<TestRuntime>,
) -> HostData {
    HostData::from_tensor_handle(client, output, HostDataType::F32)
}

pub fn validate_test(
    result: Result<(), InterpolateError>,
    actual: cubek_test_utils::HostData,
    expected: cubek_test_utils::HostData,
    tolerance: f32,
) {
    let outcome = match ExecutionOutcome::from(result) {
        ExecutionOutcome::Executed => {
            assert_equals_approx(&actual, &expected, tolerance).as_test_outcome()
        }
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    };
    outcome.enforce();
}

pub fn make_problem(
    input_shape: [usize; 4],
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> InterpolateProblem {
    InterpolateProblem {
        input_shape,
        output_size,
        options,
    }
}

pub fn run_test(
    client: ComputeClient<TestRuntime>,
    seed: u64,
    input_min: f32,
    input_max: f32,
    problem: InterpolateProblem,
    tolerance: f32,
) {
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape.to_vec())
        .uniform(seed, input_min, input_max)
        .generate_with_f32_host_data();

    let output_shape = build_output_shape(&input_data, problem.output_size);
    let output = build_output_tensor(&client, output_shape.clone(), input.dtype);
    let result = interpolate(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        problem.options.clone(),
        input.dtype,
    );

    let output_host = output_host_f32(&client, output);
    let reference = cpu_reference_from_host(&input_data, &output_shape, &problem.options, None);

    validate_test(result, output_host, reference, tolerance);
}
