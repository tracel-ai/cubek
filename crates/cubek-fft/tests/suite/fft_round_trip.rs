use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek_fft::{irfft, rfft};
//use cubefx_engine::{SignalSpec, phase_shift_effect};
use cubek_test_utils::{
    DataKind, Distribution, HostData, StrideSpec, TestInput, assert_equals_approx,
};

#[test]
fn large_fft_roundtrip() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    let signal_duration = 10.;
    let channels = 2;
    let sample_rate = 44100;
    let window_length = 2048;
    let hop_length = 1024;

    let total_samples = (signal_duration * sample_rate as f32).ceil() as usize;
    let num_windows = total_samples.div_ceil(hop_length);
    let shape = [num_windows, channels, window_length];

    let (original_signal, signal_data) = TestInput::new(
        client.clone(),
        shape,
        dtype,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 42,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    let (spectrum_re, spectrum_im) = rfft(original_signal, dtype);
    let signal_back = irfft(spectrum_re, spectrum_im, dtype);

    assert_equals_approx(
        &HostData::from_tensor_handle(&client, signal_back, cubek_test_utils::HostDataType::F32),
        &signal_data,
        0.03,
    )
    .as_test_outcome()
    .enforce();
}
