mod adaptive_avg_pool2d;
mod avg_pool2d;
mod max_pool2d;

use super::{build_output_tensor, output_host_f32, validate_test};
use cubecl::{TestRuntime, client::ComputeClient, zspace::Shape};
use cubek_pool::{
    cpu_reference::{cpu_reference_pool, geometry::PoolGeometry},
    definition::{PoolForwardProblem, PoolMode},
    pool2d,
};
use cubek_test_utils::TestInput;

pub fn make_problem(
    input_shape: Shape,
    with_indices: bool,
    mode: impl Into<PoolMode<2>>,
) -> PoolForwardProblem<2> {
    PoolForwardProblem {
        input_shape,
        with_indices,
        mode: mode.into(),
    }
}

pub fn run_pool_test(
    client: ComputeClient<TestRuntime>,
    seed: u64,
    input_min: f32,
    input_max: f32,
    problem: PoolForwardProblem<2>,
    tolerance: f32,
) {
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape.to_vec())
        .uniform(seed, input_min, input_max)
        .generate_with_f32_host_data();

    let output_shape = problem.output_shape(&problem.input_shape).to_vec();
    let output = build_output_tensor(&client, output_shape.clone(), input.dtype);

    let result = pool2d(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        problem.mode.clone(),
        input.dtype,
    );

    let output_host = output_host_f32(&client, output);

    let reference = cpu_reference_pool(&input_data, problem);

    validate_test(result, output_host, reference, tolerance);
}
