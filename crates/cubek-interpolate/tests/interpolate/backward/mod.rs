mod nearest_backward;

use cubecl::{TestRuntime, client::ComputeClient};
use cubek_interpolate::{
    definition::{InterpolateBackwardProblem, InterpolateOptions},
    eval::cpu_reference::cpu_reference_interpolate_backward_from_host,
    interpolate_backward,
};
use cubek_test_utils::TestInput;

use super::{build_output_tensor, output_host_f32, validate_test};

pub fn make_interpolate_backward_problem(
    input_size: [usize; 2],
    out_grad_shape: [usize; 4],
    options: InterpolateOptions,
) -> InterpolateBackwardProblem {
    InterpolateBackwardProblem {
        input_size,
        out_grad_shape,
        options,
    }
}

pub fn run_interpolate_backward_test(
    client: ComputeClient<TestRuntime>,
    seed: u64,
    input_min: f32,
    input_max: f32,
    problem: InterpolateBackwardProblem,
    tolerance: f32,
) {
    let (out_grad, out_grad_data) =
        TestInput::builder(client.clone(), problem.out_grad_shape.to_vec())
            .uniform(seed, input_min, input_max)
            .generate_with_f32_host_data();

    let input_shape = vec![
        problem.out_grad_shape[0],
        problem.input_size[0],
        problem.input_size[1],
        problem.out_grad_shape[3],
    ];
    let (input, _input_data) = TestInput::builder(client.clone(), input_shape.clone())
        .uniform(seed.wrapping_add(1), input_min, input_max)
        .generate_with_f32_host_data();

    let output_shape = input_shape;
    let output = build_output_tensor(&client, output_shape.clone(), out_grad.dtype);
    let result = interpolate_backward(
        &client,
        input.clone().binding(),
        out_grad.clone().binding(),
        output.clone().binding(),
        problem.options,
        out_grad.dtype,
    );

    let output_host = output_host_f32(&client, output);
    let reference = cpu_reference_interpolate_backward_from_host(
        &out_grad_data,
        &output_shape,
        &problem.options,
    );

    validate_test(result, output_host, reference, tolerance);
}
