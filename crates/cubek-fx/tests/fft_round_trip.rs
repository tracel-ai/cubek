use cubecl::{Runtime, TestRuntime, prelude::CubePrimitive};
use cubek_fx::{irfft, rfft};
//use cubefx_engine::{SignalSpec, phase_shift_effect};
use cubek_test_utils::{
    DataKind, Distribution, HostData, StrideSpec, TestInput, assert_equals_approx,
};

#[test]
fn large_fft_roundtrip_no_phase_shift() {
    // Assumptions for the benchmarks and tests:
    // - channels will be 1 or 2
    // - sample_rate won't change
    // - window_length will always be a large power of 2
    // - hop_length always equals window_length / 2
    //let signal_spec = SignalSpec {
    //    signal_duration: 10.,
    //    channels: 2,
    //    sample_rate: 44100,
    //    window_length: 2048,
    //    hop_length: 1024,
    //};

    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    let signal_duration = 10.;
    let channels = 2;
    let sample_rate = 44100;
    let window_length = 2048;
    let hop_length = 1024;

    let total_samples = (signal_duration * sample_rate as f32).ceil() as usize;
    let num_windows = total_samples.div_ceil(hop_length);
    let num_frequency_bins = window_length / 2 + 1;
    let shape = [num_windows, channels, num_frequency_bins];

    let _seed = 42;

    let (original_signal, signal_data) = TestInput::new(
        client.clone(),
        //signal_spec.signal_shape().to_vec(),
        shape,
        dtype,
        StrideSpec::RowMajor,
        DataKind::Random {
            seed: 42,
            distribution: Distribution::Uniform(-1., 1.),
        },
    )
    .generate_with_f32_host_data();

    // No phase shift, should get original signal back
    //let signal_back = phase_shift_effect(original_signal, 0., dtype);
    let (spectrum_re, spectrum_im) = rfft(original_signal, dtype);
    let signal_back = irfft(spectrum_re, spectrum_im, dtype);

    assert_equals_approx(
        &HostData::from_tensor_handle(&client, signal_back, cubek_test_utils::HostDataType::F32),
        &signal_data,
        0.01,
    )
    .as_test_outcome()
    .enforce();
}

#[test]
fn small_fft_round_trip_with_phase_shift() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();
    let num_windows = 1;
    let num_channels = 1;
    let window_length = 16;
    let shape = [num_windows, num_channels, window_length];

    let original_signal = TestInput::new(
        client.clone(),
        shape.to_vec(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: [
                0.0, 0.841, 1.207, 1.207, 0.841, 0.0, -0.841, -1.207, -1.207, -0.841, 0.0, 0.841,
                1.207, 1.207, 0.841, 0.0,
            ]
            .to_vec(),
        },
    )
    .generate_without_host_data();
    //let signal_back = phase_shift_effect(original_signal, 3.14, dtype);
    let (spectrum_re, spectrum_im) = rfft(original_signal, dtype);
    let signal_back = irfft(spectrum_re, spectrum_im, dtype);

    let (_, expected) = TestInput::new(
        client.clone(),
        shape.to_vec(),
        dtype,
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: [
                -1.2075438,
                -0.84357643,
                -0.0038496852,
                0.838393,
                1.2065021,
                1.2074304,
                0.84386146,
                0.0027349591,
                -0.0027182102,
                0.8381268,
                1.2065642,
                1.2074916,
                0.843598,
                0.0038497448,
                -0.83841383,
                -1.20645,
            ]
            .to_vec(),
        },
    )
    .generate_with_f32_host_data();

    assert_equals_approx(
        &HostData::from_tensor_handle(&client, signal_back, cubek_test_utils::HostDataType::F32),
        &expected,
        0.01,
    )
    .as_test_outcome()
    .enforce();
}
