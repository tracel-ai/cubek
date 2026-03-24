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

pub struct SignalSpec {
    pub signal_duration: f32,
    pub channels: usize,
    pub sample_rate: usize,
    pub window_length: usize,
    pub hop_length: usize,
}

impl SignalSpec {
    pub fn signal_shape(&self) -> [usize; 3] {
        let total_samples = (self.signal_duration * self.sample_rate as f32).ceil() as usize;
        let num_windows = total_samples.div_ceil(self.hop_length);
        [num_windows, self.channels, self.window_length]
    }

    pub fn spectrum_shape(&self) -> [usize; 3] {
        let total_samples = (self.signal_duration * self.sample_rate as f32).ceil() as usize;
        let num_windows = total_samples.div_ceil(self.hop_length);
        let num_frequency_bins = self.window_length / 2 + 1;
        [num_windows, self.channels, num_frequency_bins]
    }
}

fn test_launch(client: ComputeClient<TestRuntime>, signal_spec: SignalSpec) {
    let signal_shape = signal_spec.signal_shape();
    let spectrum_shape = signal_spec.spectrum_shape();

    let dtype = f32::as_type_native_unchecked().storage_type();

    let (random_spectrum_re_handle, random_spectrum_re_data) = TestInput::new(
        client.clone(),
        spectrum_shape.to_vec(),
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
        spectrum_shape.to_vec(),
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
        signal_shape.to_vec(),
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

    let signal_spec = SignalSpec {
        signal_duration: 0.1,
        channels: 2,
        sample_rate: 44100,
        window_length: 2048,
        hop_length: 1024,
    };

    test_launch(client, signal_spec);
}

#[test]
fn mono_500ms() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let signal_spec = SignalSpec {
        signal_duration: 0.5,
        channels: 1,
        sample_rate: 44100,
        window_length: 2048,
        hop_length: 1024,
    };

    test_launch(client, signal_spec);
}
