use cubecl::{
    client::ComputeClient,
    frontend::CubePrimitive,
    std::tensor::TensorHandle,
    {Runtime, TestRuntime},
};
use cubek_fft::rfft_launch;
use cubek_test_utils::{
    self, ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, ValidationResult,
    assert_equals_approx,
};

use crate::suite::reference::rfft_ref;

fn test_launch(client: ComputeClient<TestRuntime>, signal_shape: Vec<usize>, dim: usize) {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let mut spectrum_shape = signal_shape.clone();
    spectrum_shape[dim] = signal_shape[dim] / 2 + 1;

    let (white_noise_handle, white_noise_data) =
        TestInput::builder(client.clone(), signal_shape.clone())
            .dtype(dtype)
            .uniform(42, -1., 1.)
            .generate_with_f32_host_data();

    let spectrum_re_handle = TestInput::builder(client.clone(), spectrum_shape.to_vec())
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    let spectrum_im_handle = TestInput::builder(client.clone(), spectrum_shape.to_vec())
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    match rfft_launch::<TestRuntime>(
        &client,
        white_noise_handle.binding(),
        spectrum_re_handle.clone().binding(),
        spectrum_im_handle.clone().binding(),
        dim,
        dtype,
    )
    .into()
    {
        ExecutionOutcome::Executed => assert_rfft_result(
            &client,
            white_noise_data,
            spectrum_re_handle,
            spectrum_im_handle,
            dim,
        )
        .as_test_outcome(),
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce();
}

pub fn assert_rfft_result(
    client: &ComputeClient<TestRuntime>,
    signal: HostData,
    spectrum_re: TensorHandle<TestRuntime>,
    spectrum_im: TensorHandle<TestRuntime>,
    dim: usize,
) -> ValidationResult {
    // big epsilon because with wgpu, compute is less precise
    let epsilon = 0.4;
    let (expected_re, expected_im) = rfft_ref(&signal, dim);

    let actual_spectrum_re = HostData::from_tensor_handle(client, spectrum_re, HostDataType::F32);
    let actual_spectrum_im = HostData::from_tensor_handle(client, spectrum_im, HostDataType::F32);

    let result_spectrum_re = assert_equals_approx(&actual_spectrum_re, &expected_re, epsilon);
    let result_spectrum_im = assert_equals_approx(&actual_spectrum_im, &expected_im, epsilon);

    use ValidationResult::*;
    match (result_spectrum_re, result_spectrum_im) {
        (Fail(e), _) | (_, Fail(e)) => Fail(e.clone()),
        (Skipped(r1), Skipped(r2)) => Skipped(format!("{}, {}", r1, r2)),
        (Skipped(r), Pass) | (Pass, Skipped(r)) => Skipped(r.clone()),
        (Pass, Pass) => Pass,
        _ => panic!("unreachable"),
    }
}

#[test]
fn rfft_3d_axis_last() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [5, 2, 2048].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_3d_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [5, 64, 1000].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_3d_axis_0_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [128, 6, 1000].to_vec();
    let dim = 0;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_4d_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [5, 256, 6, 42].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_3d_batch_singleton_dim() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [22, 1, 2048].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}
