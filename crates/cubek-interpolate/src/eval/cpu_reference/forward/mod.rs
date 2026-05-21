mod bicubic;
mod bilinear;
mod lanczos3;
mod nearest;

pub(crate) use bicubic::reference_bicubic;
pub(crate) use bilinear::reference_bilinear;
pub(crate) use lanczos3::reference_lanczos3;
pub(crate) use nearest::reference_nearest;

use super::{f32_storage_type, make_random_f32_host, make_zero_handle, output_shape_for};
use crate::definition::{InterpolateForwardProblem, InterpolateMode, InterpolateOptions};
use cubecl::{TestRuntime, client::ComputeClient};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, Progress, launch_and_capture_outcome,
};

use crate::interpolate;

pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: InterpolateForwardProblem,
    seed: u64,
) -> Result<HostData, String> {
    let dtype = f32_storage_type();
    let input_shape = problem.input_shape.to_vec();
    let (input_handle, _input_host) = make_random_f32_host(&client, input_shape.clone(), seed);

    let out_shape = output_shape_for(&problem.input_shape, &problem.output_size);
    let output_handle = make_zero_handle(&client, out_shape, dtype);

    let outcome = launch_and_capture_outcome(&client, |c| {
        interpolate::<TestRuntime>(
            c,
            input_handle.clone().binding(),
            output_handle.clone().binding(),
            problem.options,
            dtype,
        )
        .into()
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
    problem: InterpolateForwardProblem,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    let out_shape = output_shape_for(&problem.input_shape, &problem.output_size);

    if let Some(p) = progress {
        let total: usize = out_shape.iter().product();
        p.set_total(total as u64);
    }

    let (_input_handle, input_host) =
        make_random_f32_host(&client, problem.input_shape.to_vec(), seed);

    Ok(reference_for_interpolation_mode(
        &input_host,
        &out_shape,
        &problem.options,
        progress,
    ))
}

pub fn reference_for_interpolation_mode(
    input: &HostData,
    output_shape: &[usize],
    options: &InterpolateOptions,
    progress: Option<&Progress>,
) -> HostData {
    match options.mode {
        InterpolateMode::Nearest(nearest_mode) => {
            reference_nearest(input, output_shape, nearest_mode, progress)
        }
        InterpolateMode::Bilinear => {
            reference_bilinear(input, output_shape, options.align_corners, progress)
        }
        InterpolateMode::Bicubic => {
            reference_bicubic(input, output_shape, options.align_corners, progress)
        }
        InterpolateMode::Lanczos3 => {
            reference_lanczos3(input, output_shape, options.align_corners, progress)
        }
    }
}
