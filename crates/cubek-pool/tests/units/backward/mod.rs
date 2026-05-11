mod adaptive_avg_pool2d;
mod avg_pool2d;
mod max_pool2d;

use super::{
    build_output_tensor, indices_storage_type, output_host_f32, output_host_i32, validate_indices,
    validate_test,
};
use cubecl::{TestRuntime, client::ComputeClient, zspace::Shape};
use cubek_pool::{
    cpu_reference::{cpu_reference_max_pool_indices, cpu_reference_pool_backward},
    definition::{PoolBackwardProblem, PoolMode},
    pool2d_backward, pool2d_with_indices_backward,
};
use cubek_test_utils::TestInput;

pub fn make_problem(
    input_size: [usize; 2],
    out_grad_shape: Shape,
    with_indices: bool,
    mode: impl Into<PoolMode<2>>,
) -> PoolBackwardProblem<2> {
    PoolBackwardProblem {
        input_size,
        out_grad_shape,
        with_indices,
        mode: mode.into(),
    }
}

pub fn run_pool_backward_test(
    client: ComputeClient<TestRuntime>,
    seed: u64,
    input_min: f32,
    input_max: f32,
    problem: PoolBackwardProblem<2>,
    tolerance: f32,
) {
    let input_shape = vec![
        problem.out_grad_shape[0],
        problem.input_size[0],
        problem.input_size[1],
        problem.out_grad_shape[3],
    ];
    let (input, input_data) = TestInput::builder(client.clone(), input_shape.clone())
        .uniform(seed, input_min, input_max)
        .generate_with_f32_host_data();

    let (out_grad, out_grad_data) =
        TestInput::builder(client.clone(), problem.out_grad_shape.to_vec())
            .uniform(seed + 1, input_min, input_max)
            .generate_with_f32_host_data();

    let in_grad = build_output_tensor(&client, input_shape, input.dtype);
    let indices = if problem.with_indices {
        Some(build_output_tensor(
            &client,
            problem.out_grad_shape.to_vec(),
            indices_storage_type(),
        ))
    } else {
        None
    };

    let result = if problem.with_indices {
        let indices_handle = indices.as_ref().expect("indices tensor missing");
        let pool_output =
            build_output_tensor(&client, problem.out_grad_shape.to_vec(), input.dtype);

        let _ = cubek_pool::pool2d_with_indices(
            &client,
            input.clone().binding(),
            pool_output.binding(),
            indices_handle.clone().binding(),
            problem.mode.clone(),
            input.dtype,
        );

        pool2d_with_indices_backward(
            &client,
            input.clone().binding(),
            out_grad.binding(),
            indices_handle.clone().binding(),
            in_grad.clone().binding(),
            problem.mode.clone(),
            input.dtype,
            indices_handle.dtype,
        )
    } else {
        pool2d_backward(
            &client,
            input.clone().binding(),
            out_grad.binding(),
            in_grad.clone().binding(),
            problem.mode.clone(),
            input.dtype,
        )
    };

    let in_grad_host = output_host_f32(&client, in_grad);
    let indices_host = if problem.with_indices {
        Some(output_host_i32(
            &client,
            indices.expect("indices tensor missing"),
        ))
    } else {
        None
    };

    if problem.with_indices {
        if let PoolMode::Max(opts) = &problem.mode {
            let indices_reference =
                cpu_reference_max_pool_indices(&input_data, opts, &problem.out_grad_shape.to_vec());
            validate_indices(
                indices_host
                    .as_ref()
                    .expect("indices host data missing")
                    .clone(),
                indices_reference,
            );
        }
    }

    let reference = if problem.with_indices {
        cpu_reference_pool_backward(
            &out_grad_data,
            indices_host.as_ref().expect("indices host data missing"),
            problem,
        )
    } else {
        cpu_reference_pool_backward(&out_grad_data, &input_data, problem)
    };

    validate_test(result, in_grad_host, reference, tolerance);
}
