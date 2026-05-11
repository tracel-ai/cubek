mod adaptive_avg_pool;
mod avg_pool;
mod max_pool;

pub use adaptive_avg_pool::run_adaptive_avg_pool;
pub use avg_pool::run_avg_pool;
pub use max_pool::{run_max_pool, run_max_pool_with_indices};

use super::{f32_storage_type, i32_storage_type, make_random_f32_host, make_zero_handle};
use crate::definition::{PoolForwardProblem, PoolMode};
use crate::eval::cpu_reference::{cpu_reference_pool, decode_index, geometry::PoolGeometry};
use crate::{pool2d, pool2d_with_indices};
use cubecl::{TestRuntime, client::ComputeClient};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, Progress, launch_and_capture_outcome,
};

pub(crate) fn get_window_coords<const N: usize>(
    spatial_out: &[usize],
    k_coords: &[usize],
    stride: [usize; N],
    padding: [usize; N],
    dilation: [usize; N],
    in_dims: &[usize],
    mut in_coords: Vec<usize>,
) -> Option<Vec<usize>> {
    for d in 0..N {
        let id = spatial_out[d] * stride[d] + k_coords[d] * dilation[d];
        let id_signed = id as isize - padding[d] as isize;

        if id_signed < 0 || id_signed >= in_dims[d + 1] as isize {
            return None;
        }
        in_coords[d + 1] = id_signed as usize;
    }
    Some(in_coords)
}

pub(crate) fn decode_index_simple(index: usize, shape: &[usize]) -> Vec<usize> {
    let strides = row_major_strides_vec(shape);
    decode_index(index, shape, &strides)
}

pub(crate) fn row_major_strides_vec(shape: &[usize]) -> Vec<usize> {
    let mut strides = vec![1; shape.len()];
    for i in (0..shape.len() - 1).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: PoolForwardProblem<2>,
    seed: u64,
) -> Result<HostData, String> {
    let dtype = f32_storage_type();
    let (input_handle, _input_host) =
        make_random_f32_host(&client, problem.input_shape.to_vec(), seed);

    let output_shape = problem.output_shape(&problem.input_shape);
    let output_handle = make_zero_handle(&client, output_shape.to_vec(), dtype);

    let outcome = launch_and_capture_outcome(&client, |c| {
        if problem.with_indices {
            if !matches!(&problem.mode, PoolMode::Max(_)) {
                return Err("pool indices only supported for max".to_string()).into();
            }

            let indices_handle =
                make_zero_handle(&client, output_shape.to_vec(), i32_storage_type());

            pool2d_with_indices::<TestRuntime>(
                c,
                input_handle.clone().binding(),
                output_handle.clone().binding(),
                indices_handle.clone().binding(),
                problem.mode.clone(),
                dtype,
            )
            .into()
        } else {
            pool2d::<TestRuntime>(
                c,
                input_handle.clone().binding(),
                output_handle.clone().binding(),
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
            output_handle,
            HostDataType::F32,
        )),
    }
}

pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    problem: PoolForwardProblem<2>,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    let output_shape = problem.output_shape(&problem.input_shape);

    if let Some(p) = progress {
        let total: usize = output_shape.iter().product();
        p.set_total(total as u64);
    }

    let (_input_handle, input_host) =
        make_random_f32_host(&client, problem.input_shape.to_vec(), seed);

    Ok(cpu_reference_pool(&input_host, problem))
}
