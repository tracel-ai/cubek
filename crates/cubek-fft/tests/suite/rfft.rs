#[cfg(feature = "heavy")]
use cubecl::CubeElement;
use cubecl::{
    client::ComputeClient,
    frontend::CubePrimitive,
    std::tensor::TensorHandle,
    {Runtime, TestRuntime},
};
use cubek_fft::rfft_launch;
#[cfg(feature = "heavy")]
use cubek_test_utils::HostDataVec;
use cubek_test_utils::{
    self, ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, ValidationResult,
    assert_equals_approx,
};

use cubek_fft::cpu_reference::rfft_ref;

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
    let (expected_re, expected_im) = rfft_ref(&signal, dim, None);

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

#[cfg(feature = "heavy")]
fn to_f32(host: HostData) -> Vec<f32> {
    match host.data {
        HostDataVec::F32(v) => v,
        _ => panic!("expected f32 host data"),
    }
}

#[test]
#[cfg(not(feature = "heavy"))]
fn rfft_light_axis_last() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [1, 8].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(not(feature = "heavy"))]
fn rfft_light_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [2, 8, 1].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(not(feature = "heavy"))]
fn rfft_light_axis_1_strided_trailing_batch() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [3, 8, 2].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(not(feature = "heavy"))]
fn rfft_light_axis_0_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [8, 2].to_vec();
    let dim = 0;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(not(feature = "heavy"))]
fn rfft_light_axis_last_n16() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [1, 16].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_axis_last() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [5, 2, 2048].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [5, 64, 1000].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_axis_0_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [128, 6, 1000].to_vec();
    let dim = 0;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_4d_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [5, 256, 6, 42].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_shared_memory_cap_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [1, 4096, 1].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_large_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [1, 8192, 1].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_four_step_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [1, 16384, 1].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_batched_large_axis_last() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [3, 8192].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_nyquist_bin_large_sizes() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let dtype = f32::as_type_native_unchecked().storage_type();

    for &n_fft in &[8192usize, 16384] {
        let batch = 2;
        let n_freq = n_fft / 2 + 1;
        let signal_shape = [batch, n_fft].to_vec();
        let spectrum_shape = [batch, n_freq].to_vec();

        let signal_data: Vec<f32> = (0..batch)
            .flat_map(|_| (0..n_fft).map(|i| if i % 2 == 0 { 1.0 } else { -1.0 }))
            .collect();
        let signal_handle = client.create_from_slice(f32::as_bytes(&signal_data));
        let signal =
            TensorHandle::<TestRuntime>::new_contiguous(signal_shape, signal_handle, dtype);
        let spectrum_re = TestInput::builder(client.clone(), spectrum_shape.clone())
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();
        let spectrum_im = TestInput::builder(client.clone(), spectrum_shape)
            .dtype(dtype)
            .zeros()
            .generate_without_host_data();

        rfft_launch::<TestRuntime>(
            &client,
            signal.binding(),
            spectrum_re.clone().binding(),
            spectrum_im.clone().binding(),
            1,
            dtype,
        )
        .unwrap();

        let re = to_f32(HostData::from_tensor_handle(
            &client,
            spectrum_re,
            HostDataType::F32,
        ));
        let im = to_f32(HostData::from_tensor_handle(
            &client,
            spectrum_im,
            HostDataType::F32,
        ));

        for b in 0..batch {
            let base = b * n_freq;
            for k in 0..n_freq {
                let expected = if k == n_fft / 2 { n_fft as f32 } else { 0.0 };
                assert!(
                    (re[base + k] - expected).abs() < 1.0,
                    "n_fft={n_fft}, batch={b}, bin={k}: real={}, want {expected}",
                    re[base + k],
                );
                assert!(
                    im[base + k].abs() < 1.0,
                    "n_fft={n_fft}, batch={b}, bin={k}: imag={}",
                    im[base + k],
                );
            }
        }
    }
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_3d_batch_singleton_dim() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [22, 1, 2048].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn rfft_dispatch_more_than_wgpu_x_axis_limit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [65_536, 2].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}
