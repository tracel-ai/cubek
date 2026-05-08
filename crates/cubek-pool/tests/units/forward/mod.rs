mod max_pool2d;

use super::{build_output_tensor, output_host_f32, validate_test};
use cubecl::zspace::Shape;
use cubecl::{TestRuntime, client::ComputeClient};
use cubek_pool::{
    cpu_reference::forward::cpu_reference_max_pool2d,
    definition::{MaxPoolOptions, PoolForwardProblem, PoolMode, SlidingWindow},
    pool2d,
};
use cubek_test_utils::TestInput;

pub fn make_max_pool2d_problem(
    input_size: [usize; 2],
    kernel_size: [usize; 2],
    stride: [usize; 2],
    padding: [usize; 2],
    dilation: [usize; 2],
    ceil_mode: [bool; 2],
) -> PoolForwardProblem<2> {
    PoolForwardProblem {
        input_size,
        mode: PoolMode::Max(MaxPoolOptions::<2> {
            window: SlidingWindow::<2> {
                kernel_size,
                stride,
                padding,
                ceil_mode,
            },
            dilation,
        }),
    }
}

pub fn run_max_pool2d_test(
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

    let max_config = match &problem.mode {
        PoolMode::Max(config) => config,
        _ => panic!("max pool test received non-max config"),
    };

    let output_shape = problem.mode.output_shape(&problem.input_shape);
    let output = build_output_tensor(&client, output_shape.clone(), input.dtype);

    let result = pool2d(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        problem.mode.clone(),
        input.dtype,
    );

    let output_host = output_host_f32(&client, output);
    let out_shape_arr: [usize; 4] = [
        output_shape[0],
        output_shape[1],
        output_shape[2],
        output_shape[3],
    ];

    let reference = cpu_reference_max_pool2d(
        &input_data,
        problem.input_shape,
        out_shape_arr,
        max_config.window.kernel_size,
        max_config.window.stride,
        max_config.window.padding,
        max_config.dilation,
    );

    validate_test(result, output_host, reference, tolerance);
}
