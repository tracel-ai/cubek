mod bicubic;
mod bilinear;
mod lanczos3;
mod nearest;

use cubecl::{TestRuntime, client::ComputeClient};
use cubek_interpolate::{
    definition::{InterpolateForwardProblem, InterpolateOptions},
    eval::cpu_reference::cpu_reference_interpolate_from_host,
    interpolate,
    launch::InterpolateStrategy,
    routines::{BlueprintStrategy, GlobalMemoryRoutine, GlobalMemoryStrategy},
};
use cubek_test_utils::TestInput;

use super::{build_output_tensor, output_host_f32, validate_test};

pub fn make_problem(
    input_shape: [usize; 4],
    output_size: [usize; 2],
    options: InterpolateOptions,
) -> InterpolateForwardProblem {
    InterpolateForwardProblem::from_input_output_shapes(
        &input_shape.into(),
        &output_size.into(),
        options,
    )
}

pub fn run_interpolate_global_test(
    client: ComputeClient<TestRuntime>,
    seed: u64,
    input_min: f32,
    input_max: f32,
    problem: InterpolateForwardProblem,
    tolerance: f32,
) {
    let (input, input_data) = TestInput::builder(client.clone(), &problem.input_shape())
        .uniform(seed, input_min, input_max)
        .generate_with_f32_host_data();

    let reference =
        cpu_reference_interpolate_from_host(&input_data, &problem.output_shape(), &problem.options);

    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);

    let result = interpolate(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        problem.options.clone(),
        InterpolateStrategy::GlobalMemoryStrategy(
            BlueprintStrategy::<GlobalMemoryRoutine>::Inferred(GlobalMemoryStrategy {}),
        ),
        input.dtype,
    );

    let output_host = output_host_f32(&client, output);
    validate_test(result, output_host, reference.clone(), tolerance);
}
