mod nearest;

pub(crate) use nearest::reference_nearest_backward;

use super::{f32_storage_type, make_random_f32_host, make_zero_handle};
use crate::definition::{InterpolateBackwardProblem, InterpolateMode, InterpolateOptions};
use cubecl::{TestRuntime, client::ComputeClient};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, Progress, launch_and_capture_outcome,
};

use crate::interpolate_backward;

pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: InterpolateBackwardProblem,
    seed: u64,
) -> Result<HostData, String> {
    let dtype = f32_storage_type();

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

    let input_grad_shape = input_shape;
    let input_grad_handle = make_zero_handle(&client, input_grad_shape, dtype);

    let outcome = launch_and_capture_outcome(&client, |c| {
        interpolate_backward(
            c,
            input_handle.clone().binding(),
            out_grad_handle.clone().binding(),
            input_grad_handle.clone().binding(),
            problem.options,
            dtype,
        )
        .into()
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
    problem: InterpolateBackwardProblem,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    let out_grad_shape = problem.out_grad_shape.to_vec();

    if let Some(p) = progress {
        let total: usize = out_grad_shape.iter().product();
        p.set_total(total as u64);
    }

    let (_out_grad_handle, out_grad_host) = make_random_f32_host(&client, out_grad_shape, seed);

    let input_grad_shape = vec![
        problem.out_grad_shape[0],
        problem.input_size[0],
        problem.input_size[1],
        problem.out_grad_shape[3],
    ];

    Ok(reference_for_backward_interpolation_mode(
        &out_grad_host,
        &input_grad_shape,
        &problem.options,
        progress,
    ))
}

pub fn reference_for_backward_interpolation_mode(
    out_grad: &HostData,
    output_shape: &[usize],
    options: &InterpolateOptions,
    progress: Option<&Progress>,
) -> HostData {
    match options.mode {
        InterpolateMode::Nearest(_) => {
            reference_nearest_backward(out_grad, output_shape, options.align_corners, progress)
        }
        InterpolateMode::Bilinear => {
            panic!("Bilinear interpolation backward is not supported by CPU reference")
        }
        InterpolateMode::Bicubic => {
            panic!("Bicubic interpolation backward is not supported by CPU reference")
        }
        InterpolateMode::Lanczos3 => {
            panic!("Lanczos3 interpolation backward is not supported by CPU reference")
        }
    }
}
