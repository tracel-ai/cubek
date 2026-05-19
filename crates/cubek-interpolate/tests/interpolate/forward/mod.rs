mod bicubic;
mod bilinear;
mod lanczos3;
mod nearest;

use cubecl::{TestRuntime, client::ComputeClient};
use cubek_interpolate::{
    definition::{InterpolateForwardProblem, InterpolateOptions},
    eval::cpu_reference::cpu_reference_interpolate_from_host,
    interpolate,
};
use cubek_test_utils::TestInput;

use super::{build_output_tensor, output_host_f32, validate_test};

pub fn make_problem(
    input_shape: [usize; 4],
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> InterpolateForwardProblem {
    InterpolateForwardProblem {
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
    problem: InterpolateForwardProblem,
    tolerance: f32,
) {
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape.to_vec())
        .uniform(seed, input_min, input_max)
        .generate_with_f32_host_data();

    let output_shape = vec![
        problem.input_shape[0],
        problem.output_size[0],
        problem.output_size[1],
        problem.input_shape[3],
    ];
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
        cpu_reference_interpolate_from_host(&input_data, &output_shape, &problem.options);

    validate_test(result, output_host, reference, tolerance);
}
