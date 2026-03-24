use cubecl::client::ComputeClient;
use cubecl::frontend::CubePrimitive;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, TestRuntime};
use cubek_fft::irfft_launch;
use cubek_test_utils::{
    self, DataKind, Distribution, ExecutionOutcome, HostData, HostDataType, StrideSpec, TestInput,
    TestOutcome, ValidationResult, assert_equals_approx,
};

use crate::suite::reference::irfft_ref;

fn test_launch(
    client: ComputeClient<TestRuntime>,
    signal_shape: Vec<usize>,
    spectrum_shape: Vec<usize>,
) {
    let dtype = f32::as_type_native_unchecked().storage_type();

    let (random_spectrum_re_handle, random_spectrum_re_data) = TestInput::new(
        client.clone(),
        spectrum_shape.clone(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 43,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (random_spectrum_im_handle, random_spectrum_im_data) = TestInput::new(
        client.clone(),
        spectrum_shape,
        dtype,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 44,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let signal_handle = TestInput::new(
        client.clone(),
        signal_shape,
        dtype,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate_without_host_data();

    match irfft_launch::<TestRuntime>(
        &client,
        random_spectrum_re_handle.binding(),
        random_spectrum_im_handle.binding(),
        signal_handle.clone().binding(),
        dtype,
    )
    .into()
    {
        ExecutionOutcome::Executed => assert_irfft_result(
            &client,
            random_spectrum_re_data,
            random_spectrum_im_data,
            signal_handle,
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
) -> ValidationResult {
    let epsilon = 0.01;
    let expected_signal = irfft_ref(&spectrum_re, &spectrum_im);
    let actual_signal = HostData::from_tensor_handle(client, signal, HostDataType::F32);

    assert_equals_approx(&actual_signal, &expected_signal, epsilon)
}

#[test]
fn stereo_100ms() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [5, 2, 2048].to_vec();
    let spectrum_shape = [5, 2, 1025].to_vec();
    test_launch(client, signal_shape, spectrum_shape);
}

#[test]
fn mono_500ms() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [22, 1, 2048].to_vec();
    let spectrum_shape = [22, 1, 1025].to_vec();
    test_launch(client, signal_shape, spectrum_shape);
}
