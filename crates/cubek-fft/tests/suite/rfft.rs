use cubecl::client::ComputeClient;
use cubecl::frontend::CubePrimitive;
use cubecl::std::tensor::TensorHandle;
use cubecl::{Runtime, TestRuntime};
use cubek_fft::rfft_launch;
use cubek_test_utils::{
    self, DataKind, Distribution, ExecutionOutcome, HostData, HostDataType, StrideSpec, TestInput,
    TestOutcome, ValidationResult, assert_equals_approx,
};

use crate::suite::reference::rfft_ref;

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

    let (white_noise_handle, white_noise_data) = TestInput::new(
        client.clone(),
        signal_shape.to_vec(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 42,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let spectrum_re_handle = TestInput::new(
        client.clone(),
        spectrum_shape.to_vec(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate_without_host_data();

    let spectrum_im_handle = TestInput::new(
        client.clone(),
        spectrum_shape.to_vec(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Zeros,
    )
    .generate_without_host_data();

    match rfft_launch::<TestRuntime>(
        &client,
        white_noise_handle.binding(),
        spectrum_re_handle.clone().binding(),
        spectrum_im_handle.clone().binding(),
        dtype,
    )
    .into()
    {
        ExecutionOutcome::Executed => assert_rfft_result(
            &client,
            white_noise_data,
            spectrum_re_handle,
            spectrum_im_handle,
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
) -> ValidationResult {
    // big epsilon because with wgpu, calculs are less precise
    let epsilon = 0.4;
    let (expected_re, expected_im) = rfft_ref(&signal);

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
