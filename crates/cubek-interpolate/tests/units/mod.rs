mod bicubic;
mod bilinear;
mod lanczos3;
mod nearest;
mod nearest_backward;

use cubecl::{TestRuntime, client::ComputeClient, ir::StorageType, std::tensor::TensorHandle};
use cubek_interpolate::cpu_reference::{
    cpu_reference_interpolate_backward_from_host, cpu_reference_interpolate_from_host,
};
use cubek_interpolate::definition::{InterpolateOptions, InterpolateProblem};
use cubek_interpolate::{InterpolateError, interpolate, interpolate_backward};
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

pub fn run_interpolate_test(
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
    let reference =
        cpu_reference_interpolate_from_host(&input_data, &output_shape, &problem.options, None);

    validate_test(result, output_host, reference, tolerance);
}

pub fn run_interpolate_backward_test(
    client: ComputeClient<TestRuntime>,
    seed: u64,
    input_min: f32,
    input_max: f32,
    problem: InterpolateProblem,
    tolerance: f32,
) {
    let out_grad_shape = vec![
        problem.input_shape[0],
        problem.output_size[0],
        problem.output_size[1],
        problem.input_shape[3],
    ];
    let (out_grad, out_grad_data) = TestInput::builder(client.clone(), out_grad_shape)
        .uniform(seed, input_min, input_max)
        .generate_with_f32_host_data();
    let (input, _input_data) = TestInput::builder(client.clone(), problem.input_shape.to_vec())
        .uniform(seed.wrapping_add(1), input_min, input_max)
        .generate_with_f32_host_data();

    let output_shape = problem.input_shape.to_vec();
    let output = build_output_tensor(&client, output_shape.clone(), out_grad.dtype);
    let result = interpolate_backward(
        &client,
        input.clone().binding(),
        out_grad.clone().binding(),
        output.clone().binding(),
        problem.options.clone(),
        out_grad.dtype,
    );

    let output_host = output_host_f32(&client, output);
    let reference = cpu_reference_interpolate_backward_from_host(
        &out_grad_data,
        &output_shape,
        &problem.options,
        None,
    );

    validate_test(result, output_host, reference, tolerance);
}
