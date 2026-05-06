use cubecl::CubeElement;
use cubecl::{
    client::ComputeClient,
    frontend::CubePrimitive,
    prelude::StorageType,
    std::tensor::TensorHandle,
    {Runtime, TestRuntime},
};
use cubek_fft::{rfft_launch, rfft_launch_padded};
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

fn test_launch_padded(
    client: ComputeClient<TestRuntime>,
    signal_shape: Vec<usize>,
    dim: usize,
    signal_len: usize,
    n_fft: usize,
) {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let n_freq = n_fft / 2 + 1;

    let mut spectrum_shape = signal_shape.clone();
    spectrum_shape[dim] = n_freq;
    let mut padded_shape = signal_shape.clone();
    padded_shape[dim] = n_fft;

    let virtual_signal = tensor_from_data(
        &client,
        signal_shape.clone(),
        &data_for_shape_with_len(&signal_shape, dim, signal_len),
        dtype,
    );
    let padded_signal = tensor_from_data(
        &client,
        padded_shape,
        &padded_data(&signal_shape, dim, signal_len, n_fft),
        dtype,
    );

    let virtual_re = empty_tensor(&client, spectrum_shape.clone(), dtype);
    let virtual_im = empty_tensor(&client, spectrum_shape.clone(), dtype);
    let padded_re = empty_tensor(&client, spectrum_shape.clone(), dtype);
    let padded_im = empty_tensor(&client, spectrum_shape, dtype);

    rfft_launch_padded::<TestRuntime>(
        &client,
        virtual_signal.binding(),
        virtual_re.clone().binding(),
        virtual_im.clone().binding(),
        dim,
        signal_len,
        dtype,
    )
    .unwrap();

    rfft_launch::<TestRuntime>(
        &client,
        padded_signal.binding(),
        padded_re.clone().binding(),
        padded_im.clone().binding(),
        dim,
        dtype,
    )
    .unwrap();

    let actual_re = to_f32(HostData::from_tensor_handle(
        &client,
        virtual_re,
        HostDataType::F32,
    ));
    let actual_im = to_f32(HostData::from_tensor_handle(
        &client,
        virtual_im,
        HostDataType::F32,
    ));
    let expected_re = to_f32(HostData::from_tensor_handle(
        &client,
        padded_re,
        HostDataType::F32,
    ));
    let expected_im = to_f32(HostData::from_tensor_handle(
        &client,
        padded_im,
        HostDataType::F32,
    ));

    assert_f32_close(&actual_re, &expected_re);
    assert_f32_close(&actual_im, &expected_im);
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

fn to_f32(host: HostData) -> Vec<f32> {
    match host.data {
        HostDataVec::F32(v) => v,
        _ => panic!("expected f32 host data"),
    }
}

fn coords_from_index(mut index: usize, shape: &[usize]) -> Vec<usize> {
    let mut coords = vec![0; shape.len()];
    for axis in (0..shape.len()).rev() {
        coords[axis] = index % shape[axis];
        index /= shape[axis];
    }
    coords
}

fn sample_value(coords: &[usize]) -> f32 {
    coords
        .iter()
        .enumerate()
        .map(|(axis, coord)| (axis as f32 + 1.0) * (*coord as f32 + 0.25))
        .sum::<f32>()
        .sin()
}

fn data_for_shape_with_len(shape: &[usize], dim: usize, signal_len: usize) -> Vec<f32> {
    (0..shape.iter().product::<usize>())
        .map(|index| {
            let coords = coords_from_index(index, shape);
            if coords[dim] < signal_len {
                sample_value(&coords)
            } else {
                0.0
            }
        })
        .collect()
}

fn padded_data(shape: &[usize], dim: usize, signal_len: usize, target_len: usize) -> Vec<f32> {
    let mut padded_shape = shape.to_vec();
    padded_shape[dim] = target_len;

    (0..padded_shape.iter().product::<usize>())
        .map(|index| {
            let coords = coords_from_index(index, &padded_shape);
            if coords[dim] < signal_len {
                sample_value(&coords)
            } else {
                0.0
            }
        })
        .collect()
}

fn tensor_from_data(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    data: &[f32],
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    TensorHandle::<TestRuntime>::new_contiguous(
        shape,
        client.create_from_slice(f32::as_bytes(data)),
        dtype,
    )
}

fn empty_tensor(
    client: &ComputeClient<TestRuntime>,
    shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    let elems = shape.iter().product::<usize>();
    TensorHandle::<TestRuntime>::new_contiguous(shape, client.empty(elems * dtype.size()), dtype)
}

fn assert_f32_close(actual: &[f32], expected: &[f32]) {
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        assert!(
            (actual - expected).abs() < 1e-4,
            "mismatch at index {index}: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn rfft_light_axis_last() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [1, 8].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_light_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [2, 8, 1].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_light_axis_1_strided_trailing_batch() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [3, 8, 2].to_vec();
    let dim = 1;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_light_axis_0_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [8, 2].to_vec();
    let dim = 0;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_light_axis_last_n16() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let signal_shape = [1, 16].to_vec();
    let dim = signal_shape.len() - 1;
    test_launch(client, signal_shape, dim);
}

#[test]
fn rfft_virtual_padding_axis_1_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_padded(client, vec![2, 5, 3], 1, 5, 8);
}

#[test]
fn rfft_virtual_padding_ignores_tail_after_signal_len() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_padded(client, vec![2, 7, 3], 1, 5, 8);
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
fn rfft_large_virtual_padding_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_padded(client, vec![1, 5000], 1, 5000, 8192);
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
