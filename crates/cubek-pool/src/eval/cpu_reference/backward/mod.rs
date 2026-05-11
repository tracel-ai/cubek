mod adaptive_avg_pool;
mod avg_pool;
mod max_pool;

pub use adaptive_avg_pool::run_adaptive_avg_pool_backward;
pub use avg_pool::run_avg_pool_backward;
pub use max_pool::run_max_pool_backward;

use super::{f32_storage_type, i32_storage_type, make_random_f32_host, make_zero_handle};
use crate::definition::{PoolBackwardProblem, PoolMode};
use crate::eval::cpu_reference::{cpu_reference_max_pool_indices, cpu_reference_pool_backward};
use crate::{pool2d_backward, pool2d_with_indices, pool2d_with_indices_backward};
use cubecl::{
    TestRuntime,
    client::ComputeClient,
    zspace::{Shape, Strides},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, Progress, launch_and_capture_outcome,
};

pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: PoolBackwardProblem<2>,
    seed: u64,
) -> Result<HostData, String> {
    if matches!(&problem.mode, PoolMode::Max(_)) && !problem.with_indices {
        return Err("max pool backward requires indices".to_string());
    }

    let dtype = f32_storage_type();
    let indices_dtype = i32_storage_type();
    let out_grad_shape = problem.out_grad_shape.to_vec();
    let input_shape = vec![
        out_grad_shape[0],
        problem.input_size[0],
        problem.input_size[1],
        out_grad_shape[3],
    ];

    let (input_handle, _input_host) = make_random_f32_host(&client, input_shape.clone(), seed);
    let (out_grad_handle, _out_grad_host) =
        make_random_f32_host(&client, out_grad_shape.clone(), seed);
    let input_grad_handle = make_zero_handle(&client, input_shape, dtype);

    let indices_handle = if problem.with_indices {
        let output_handle = make_zero_handle(&client, out_grad_shape.clone(), dtype);
        let indices_handle = make_zero_handle(&client, out_grad_shape.clone(), indices_dtype);

        let forward_outcome = launch_and_capture_outcome(&client, |c| {
            pool2d_with_indices::<TestRuntime>(
                c,
                input_handle.clone().binding(),
                output_handle.clone().binding(),
                indices_handle.clone().binding(),
                problem.mode.clone(),
                dtype,
            )
            .into()
        });

        match forward_outcome {
            ExecutionOutcome::CompileError(e) => return Err(format!("compile error: {e}")),
            ExecutionOutcome::Executed => Some(indices_handle),
        }
    } else {
        None
    };

    let outcome = launch_and_capture_outcome(&client, |c| {
        if let Some(indices) = &indices_handle {
            pool2d_with_indices_backward::<TestRuntime>(
                c,
                input_handle.clone().binding(),
                out_grad_handle.clone().binding(),
                indices.clone().binding(),
                input_grad_handle.clone().binding(),
                problem.mode.clone(),
                dtype,
                indices_dtype,
            )
            .into()
        } else {
            pool2d_backward::<TestRuntime>(
                c,
                input_handle.clone().binding(),
                out_grad_handle.clone().binding(),
                input_grad_handle.clone().binding(),
                problem.mode.clone(),
                dtype,
            )
            .into()
        }
    });

    match outcome {
        ExecutionOutcome::CompileError(e) => Err(format!("compile error: {e}")),
        ExecutionOutcome::Executed => Ok(HostData::from_tensor_handle(
            &client,
            input_grad_handle,
            HostDataType::F32,
        )),
    }
}

pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    problem: PoolBackwardProblem<2>,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    if matches!(&problem.mode, PoolMode::Max(_)) && !problem.with_indices {
        return Err("max pool backward requires indices".to_string());
    }

    let out_grad_shape = problem.out_grad_shape.to_vec();
    let input_shape = vec![
        out_grad_shape[0],
        problem.input_size[0],
        problem.input_size[1],
        out_grad_shape[3],
    ];

    if let Some(p) = progress {
        let total: usize = input_shape.iter().product();
        p.set_total(total as u64);
    }

    let (_input_handle, input_host) = make_random_f32_host(&client, input_shape, seed);
    let (_out_grad_handle, out_grad_host) =
        make_random_f32_host(&client, out_grad_shape.clone(), seed);

    let indices_host = if problem.with_indices {
        match &problem.mode {
            PoolMode::Max(opts) => {
                cpu_reference_max_pool_indices(&input_host, opts, &out_grad_shape)
            }
            _ => return Err("pool backward indices only supported for max".to_string()),
        }
    } else {
        let empty_dims: Vec<usize> = vec![];

        HostData {
            data: HostDataVec::I32(Vec::new()),
            shape: Shape::from(empty_dims),
            strides: Strides::new(&[] as &[usize]),
        }
    };

    Ok(cpu_reference_pool_backward(
        &out_grad_host,
        &indices_host,
        problem,
    ))
}
