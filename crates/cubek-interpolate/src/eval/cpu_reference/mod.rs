mod backward;
mod forward;

use crate::definition::{InterpolateOptions, InterpolateProblem};
use cubecl::ir::StorageType;
use cubecl::std::tensor::TensorHandle;
use cubecl::{TestRuntime, client::ComputeClient, prelude::*, zspace::Strides};
use cubek_test_utils::{HostData, Progress, TestInput};

pub(crate) fn f32_storage_type() -> StorageType {
    f32::as_type_native_unchecked().storage_type()
}

pub(crate) fn make_random_f32_host(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    seed: u64,
) -> (TensorHandle<TestRuntime>, HostData) {
    TestInput::builder(client.clone(), shape)
        .uniform(seed, -1., 1.)
        .generate_with_f32_host_data()
}

pub(crate) fn make_zero_handle(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), shape)
        .dtype(dtype)
        .zeros()
        .generate()
}

pub(crate) fn output_shape_for(input_shape: &[usize; 4], output_size: &[usize; 2]) -> Vec<usize> {
    let mut out = input_shape.to_vec();
    out[1] = output_size[0];
    out[2] = output_size[1];
    out
}

pub fn strategy_result(
    client: ComputeClient<TestRuntime>,
    problem: InterpolateProblem,
    seed: u64,
) -> Result<HostData, String> {
    match problem {
        InterpolateProblem::Forward(prob) => forward::strategy_result(client, prob, seed),
        InterpolateProblem::Backward(prob) => backward::strategy_result(client, prob, seed),
    }
}

pub fn cpu_reference_result(
    client: ComputeClient<TestRuntime>,
    problem: InterpolateProblem,
    seed: u64,
    progress: Option<&Progress>,
) -> Result<HostData, String> {
    match problem {
        InterpolateProblem::Forward(prob) => {
            forward::cpu_reference_result(client, prob, seed, progress)
        }
        InterpolateProblem::Backward(prob) => {
            backward::cpu_reference_result(client, prob, seed, progress)
        }
    }
}

pub fn cpu_reference_interpolate_from_host(
    input: &HostData,
    output_shape: &[usize],
    options: &InterpolateOptions,
) -> HostData {
    forward::reference_for_interpolation_mode(input, output_shape, options, None)
}

pub fn cpu_reference_interpolate_backward_from_host(
    out_grad: &HostData,
    output_shape: &[usize],
    options: &InterpolateOptions,
) -> HostData {
    backward::reference_for_backward_interpolation_mode(out_grad, output_shape, options, None)
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
