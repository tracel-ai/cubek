mod adaptive_avg_pool;
mod avg_pool;
mod max_pool;

pub use adaptive_avg_pool::run_adaptive_avg_pool;
pub use avg_pool::run_avg_pool;
pub use max_pool::{run_max_pool, run_max_pool_with_indices};

use super::{f32_elem_type, i32_elem_type, make_random_f32_host, make_zero_handle};
use crate::definition::{PoolForwardProblem, PoolMode};
use crate::eval::cpu_reference::{cpu_reference_pool, geometry::PoolGeometry};
use crate::{pool2d, pool2d_with_indices};
use cubecl::client::Client;
use cubek_test_utils::{ExecutionOutcome, HostData, HostDataType, KernelResultError, Progress};

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

/// The output of `problem`'s launch on seeded input.
///
/// # Errors
///
/// [`KernelResultError::Refused`] with the decline the launch returns, and
/// [`KernelResultError::NotRun`] where it asks indices of a pool other than max, or the launch
/// returns another error or did not run.
pub fn strategy_result(
    client: Client,
    problem: PoolForwardProblem<2>,
    seed: u64,
) -> Result<HostData, KernelResultError> {
    let dtype = f32_elem_type();
    let (input_handle, _input_host) =
        make_random_f32_host(&client, problem.input_shape.to_vec(), seed);

    let output_shape = problem.output_shape(&problem.input_shape);
    let output_handle = make_zero_handle(&client, output_shape.to_vec(), dtype);

    match problem.with_indices {
        true => {
            if !matches!(&problem.mode, PoolMode::Max(_)) {
                return Err("pool indices only supported for max".to_string().into());
            }
            let indices_handle = make_zero_handle(&client, output_shape.to_vec(), i32_elem_type());
            pool2d_with_indices(
                &client,
                input_handle.binding(),
                output_handle.clone().binding(),
                indices_handle.binding(),
                problem.mode.clone(),
                dtype,
            )?
        }
        false => pool2d(
            &client,
            input_handle.binding(),
            output_handle.clone().binding(),
            problem.mode.clone(),
            dtype,
        )?,
    }
    ExecutionOutcome::of_outputs(&client, &[&output_handle.handle]).into_kernel_result()?;
    Ok(HostData::from_tensor_handle(
        &client,
        output_handle,
        HostDataType::F32,
    ))
}

pub fn cpu_reference_result(
    client: Client,
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
