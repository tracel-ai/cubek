mod nearest;

pub use nearest::reference_nearest;

use crate::definition::{InterpolateMode, InterpolateOptions, InterpolateProblem};
use cubecl::{TestRuntime, client::ComputeClient, prelude::*, zspace::Strides};
use cubek_test_utils::{
    ExecutionOutcome, HostData, HostDataType, Progress, TestInput, launch_and_capture_outcome,
};

use crate::interpolate;

pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: InterpolateProblem,
    seed: u64,
) -> Result<HostData, String> {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let input_shape = problem.input_shape.to_vec();
    let (input_handle, _input_host) = TestInput::builder(client.clone(), input_shape.clone())
        .uniform(seed, -1., 1.)
        .generate_with_f32_host_data();

    let out_shape = output_shape_for(&problem.input_shape, &problem.output_size);
    let output_handle = TestInput::builder(client.clone(), out_shape)
        .dtype(dtype)
        .zeros()
        .generate();

    let outcome = launch_and_capture_outcome(&client, |c| {
        interpolate::<TestRuntime>(
            c,
            input_handle.clone().binding(),
            output_handle.clone().binding(),
            problem.options.clone(),
            dtype.clone(),
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
    problem: InterpolateProblem,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    let input_dtype = f32::as_type_native_unchecked().storage_type();
    let input_shape = problem.input_shape.to_vec();
    let out_shape = output_shape_for(&problem.input_shape, &problem.output_size);

    if let Some(p) = progress {
        let total: usize = out_shape.iter().product();
        p.set_total(total as u64);
    }

    let (_input_handle, input_host) = TestInput::builder(client.clone(), input_shape)
        .dtype(input_dtype)
        .uniform(seed, -1., 1.)
        .generate_with_f32_host_data();

    Ok(reference_for_mode(
        &input_host,
        &out_shape,
        &problem.options,
        progress,
    ))
}

pub fn cpu_reference_from_host(
    input: &HostData,
    output_shape: &[usize],
    options: &InterpolateOptions,
    progress: Option<&Progress>,
) -> HostData {
    reference_for_mode(input, output_shape, options, progress)
}

fn reference_for_mode(
    input: &HostData,
    output_shape: &[usize],
    options: &InterpolateOptions,
    progress: Option<&Progress>,
) -> HostData {
    match options.mode {
        InterpolateMode::Nearest => {
            reference_nearest(input, output_shape, options.align_corners, progress)
        }
        InterpolateMode::Bilinear => {
            reference_nearest(input, output_shape, options.align_corners, progress)
        }
        InterpolateMode::Bicubic => {
            reference_nearest(input, output_shape, options.align_corners, progress)
        }
        InterpolateMode::Lanczos3 => {
            reference_nearest(input, output_shape, options.align_corners, progress)
        }
    }
}

pub(crate) fn for_each_output_coord(output_shape: &[usize], mut f: impl FnMut(usize, &[usize])) {
    let rank = output_shape.len();
    if rank == 0 {
        f(0, &[]);
        return;
    }
    let num: usize = output_shape.iter().product();
    let mut coord = vec![0usize; rank];
    for linear in 0..num {
        let mut rem = linear;
        for d in (0..rank).rev() {
            coord[d] = rem % output_shape[d];
            rem /= output_shape[d];
        }
        f(linear, &coord);
    }
}

pub(crate) fn contiguous_strides(shape: &[usize]) -> Strides {
    let n = shape.len();
    if n == 0 {
        return Strides::new(&[] as &[usize]);
    }
    let mut s = vec![0usize; n];
    s[n - 1] = 1;
    for i in (0..n - 1).rev() {
        s[i] = s[i + 1] * shape[i + 1];
    }
    Strides::new(&s)
}

fn output_shape_for(input_shape: &[usize; 4], output_size: &[usize; 2]) -> Vec<usize> {
    let mut out = input_shape.to_vec();
    out[1] = output_size[0];
    out[2] = output_size[1];
    out
}
