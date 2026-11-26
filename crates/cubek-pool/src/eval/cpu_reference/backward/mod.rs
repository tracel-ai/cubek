mod adaptive_avg_pool;
mod avg_pool;
mod max_pool;

pub use adaptive_avg_pool::run_adaptive_avg_pool_backward;
pub use avg_pool::run_avg_pool_backward;
pub use max_pool::run_max_pool_backward;

use super::{f32_elem_type, i32_elem_type, make_random_f32_host, make_zero_handle};
use crate::definition::{PoolBackwardProblem, PoolMode};
use crate::eval::cpu_reference::{cpu_reference_max_pool_indices, cpu_reference_pool_backward};
use crate::{pool2d_backward, pool2d_with_indices, pool2d_with_indices_backward};
use cubecl::{
    client::Client,
    zspace::{Shape, Strides},
};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, HostDataVec, KernelResultError, Progress,
};

/// The input gradient of `problem`'s launch on seeded input and output gradient, its indices
/// computed by a forward launch where it takes them.
///
/// # Errors
///
/// [`KernelResultError::Refused`] with the decline either launch returns, and
/// [`KernelResultError::NotRun`] where a max pool takes no indices, or a launch returns another
/// error or did not run.
pub fn strategy_result(
    client: Client,
    problem: PoolBackwardProblem<2>,
    seed: u64,
) -> Result<HostData, KernelResultError> {
    if matches!(&problem.mode, PoolMode::Max(_)) && !problem.with_indices {
        return Err("max pool backward requires indices".to_string().into());
    }

    let dtype = f32_elem_type();
    let indices_dtype = i32_elem_type();
    let out_grad_shape = problem.out_grad_shape.to_vec();
    let input_shape = problem.input_shape().to_vec();

    let (input_handle, _input_host) = make_random_f32_host(&client, input_shape.clone(), seed);
    let (out_grad_handle, _out_grad_host) =
        make_random_f32_host(&client, out_grad_shape.clone(), seed);
    let input_grad_handle = make_zero_handle(&client, input_shape, dtype);

    match problem.with_indices {
        true => {
            let output_handle = make_zero_handle(&client, out_grad_shape.clone(), dtype);
            let indices_handle = make_zero_handle(&client, out_grad_shape, indices_dtype);
            pool2d_with_indices(
                &client,
                input_handle.clone().binding(),
                output_handle.clone().binding(),
                indices_handle.clone().binding(),
                problem.mode.clone(),
                dtype,
            )?;
            ExecutionOutcome::of_outputs(&client, &[&output_handle.handle, &indices_handle.handle])
                .into_kernel_result()?;
            pool2d_with_indices_backward(
                &client,
                input_handle.binding(),
                out_grad_handle.binding(),
                indices_handle.binding(),
                input_grad_handle.clone().binding(),
                problem.mode.clone(),
                dtype,
                indices_dtype,
            )?
        }
        false => pool2d_backward(
            &client,
            input_handle.binding(),
            out_grad_handle.binding(),
            input_grad_handle.clone().binding(),
            problem.mode.clone(),
            dtype,
        )?,
    }
    ExecutionOutcome::of_outputs(&client, &[&input_grad_handle.handle]).into_kernel_result()?;
    Ok(HostData::from_tensor_handle(
        &client,
        input_grad_handle,
        HostDataType::F32,
    ))
}

pub fn cpu_reference_result(
    client: Client,
    problem: PoolBackwardProblem<2>,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    if matches!(&problem.mode, PoolMode::Max(_)) && !problem.with_indices {
        return Err("max pool backward requires indices".to_string());
    }

    let out_grad_shape = problem.out_grad_shape.to_vec();
    let input_shape = problem.input_shape().to_vec();

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
