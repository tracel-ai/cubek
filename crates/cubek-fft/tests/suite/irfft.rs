use cubecl::{
    client::ComputeClient,
    frontend::CubePrimitive,
    std::tensor::TensorHandle,
    {Runtime, TestRuntime},
};
use cubek_fft::irfft_launch;
use cubek_test_utils::{
    self, ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, ValidationResult,
    assert_equals_approx,
};

use crate::suite::reference::irfft_ref;

fn test_launch(client: ComputeClient<TestRuntime>, spectrum_shape: Vec<usize>, dim: usize) {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let mut signal_shape = spectrum_shape.clone();
    signal_shape[dim] = (spectrum_shape[dim] - 1) * 2;

    let (random_spectrum_re_handle, random_spectrum_re_data) =
        TestInput::builder(client.clone(), spectrum_shape.clone())
            .dtype(dtype)
            .uniform(43, -1., 1.)
            .generate_with_f32_host_data();

    let (random_spectrum_im_handle, random_spectrum_im_data) =
        TestInput::builder(client.clone(), spectrum_shape)
            .dtype(dtype)
            .uniform(44, -1., 1.)
            .generate_with_f32_host_data();

    let signal_handle = TestInput::builder(client.clone(), signal_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data();

    match irfft_launch::<TestRuntime>(
        &client,
        random_spectrum_re_handle.binding(),
        random_spectrum_im_handle.binding(),
        signal_handle.clone().binding(),
        dim,
        dtype,
    )
    .into()
    {
        ExecutionOutcome::Executed => assert_irfft_result(
            &client,
            random_spectrum_re_data,
            random_spectrum_im_data,
            signal_handle,
            dim,
        )
        .as_test_outcome(),
        ExecutionOutcome::CompileError(e) => TestOutcome::CompileError(e),
    }
    .enforce();
}

fn assert_irfft_result(
    client: &ComputeClient<TestRuntime>,
    spectrum_re: HostData,
    spectrum_im: HostData,
    signal: TensorHandle<TestRuntime>,
    dim: usize,
) -> ValidationResult {
    let epsilon = 0.01;
    let expected_signal = irfft_ref(&spectrum_re, &spectrum_im, dim);
    let actual_signal = HostData::from_tensor_handle(client, signal, HostDataType::F32);

    assert_equals_approx(&actual_signal, &expected_signal, epsilon)
}

#[test]
fn irfft_3d_last_axis() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [5, 2, 1025].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_3d_axis_0() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [33, 2, 1024].to_vec();
    let dim = 0;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_3d_axis_1() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [33, 5, 1024].to_vec();
    let dim = 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_4d_axis_2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [12, 8, 513, 4].to_vec();
    let dim = 2;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_3d_batch_singleton_dim() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [22, 1, 1025].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_dispatch_more_than_wgpu_x_axis_limit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [65_536, 2].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}
