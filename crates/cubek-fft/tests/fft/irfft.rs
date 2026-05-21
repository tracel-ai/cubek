use cubecl::CubeElement;
use cubecl::{
    client::ComputeClient,
    frontend::CubePrimitive,
    prelude::StorageType,
    std::tensor::TensorHandle,
    {Runtime, TestRuntime},
};
use cubek_fft::{irfft_launch, irfft_launch_padded};
use cubek_test_utils::{
    self, ExecutionOutcome, HostData, HostDataType, TestInput, TestOutcome, ValidationResult,
    assert_equals_approx, launch_and_capture_outcome,
};

use cubek_fft::eval::cpu_reference::irfft_ref;

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

    let re_binding = random_spectrum_re_handle.binding();
    let im_binding = random_spectrum_im_handle.binding();
    let signal_binding = signal_handle.clone().binding();

    let outcome = launch_and_capture_outcome(&client, |c| {
        irfft_launch::<TestRuntime>(c, re_binding, im_binding, signal_binding, dim, dtype).into()
    });

    match outcome {
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

fn test_launch_padded(
    client: ComputeClient<TestRuntime>,
    spectrum_shape: Vec<usize>,
    dim: usize,
    n_fft: usize,
) {
    let dtype = f32::as_type_native_unchecked().storage_type();
    let spec_bins = spectrum_shape[dim];
    let n_freq = n_fft / 2 + 1;

    let mut full_spectrum_shape = spectrum_shape.clone();
    full_spectrum_shape[dim] = n_freq;
    let mut signal_shape = spectrum_shape.clone();
    signal_shape[dim] = n_fft;

    let virtual_re = tensor_from_data(
        &client,
        spectrum_shape.clone(),
        &data_for_shape(&spectrum_shape),
        dtype,
    );
    let virtual_im = tensor_from_data(
        &client,
        spectrum_shape.clone(),
        &data_for_shape(&spectrum_shape),
        dtype,
    );
    let padded_re = tensor_from_data(
        &client,
        full_spectrum_shape.clone(),
        &padded_data(&spectrum_shape, dim, n_freq),
        dtype,
    );
    let padded_im = tensor_from_data(
        &client,
        full_spectrum_shape,
        &padded_data(&spectrum_shape, dim, n_freq),
        dtype,
    );

    let virtual_signal = empty_tensor(&client, signal_shape.clone(), dtype);
    let padded_signal = empty_tensor(&client, signal_shape, dtype);

    let virtual_re_binding = virtual_re.binding();
    let virtual_im_binding = virtual_im.binding();
    let virtual_signal_binding = virtual_signal.clone().binding();
    let padded_re_binding = padded_re.binding();
    let padded_im_binding = padded_im.binding();
    let padded_signal_binding = padded_signal.clone().binding();

    let outcome = launch_and_capture_outcome(&client, |c| {
        if let Err(e) = irfft_launch_padded::<TestRuntime>(
            c,
            virtual_re_binding,
            virtual_im_binding,
            virtual_signal_binding,
            dim,
            spec_bins,
            dtype,
        ) {
            return ExecutionOutcome::CompileError(format!("virtual launch failed: {e}"));
        }
        irfft_launch::<TestRuntime>(
            c,
            padded_re_binding,
            padded_im_binding,
            padded_signal_binding,
            dim,
            dtype,
        )
        .into()
    });

    match outcome {
        ExecutionOutcome::Executed => {
            let actual = HostData::from_tensor_handle(&client, virtual_signal, HostDataType::F32);
            let expected = HostData::from_tensor_handle(&client, padded_signal, HostDataType::F32);
            assert_equals_approx(&actual, &expected, 1e-4).as_test_outcome()
        }
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
    let expected_signal = irfft_ref(&spectrum_re, &spectrum_im, dim, None);
    let actual_signal = HostData::from_tensor_handle(client, signal, HostDataType::F32);

    assert_equals_approx(&actual_signal, &expected_signal, epsilon)
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

fn data_for_shape(shape: &[usize]) -> Vec<f32> {
    (0..shape.iter().product::<usize>())
        .map(|index| sample_value(&coords_from_index(index, shape)))
        .collect()
}

fn padded_data(shape: &[usize], dim: usize, target_len: usize) -> Vec<f32> {
    let mut padded_shape = shape.to_vec();
    padded_shape[dim] = target_len;

    (0..padded_shape.iter().product::<usize>())
        .map(|index| {
            let coords = coords_from_index(index, &padded_shape);
            if coords[dim] < shape[dim] {
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

#[test]
fn irfft_light_axis_last() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [1, 5].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_light_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [2, 5, 1].to_vec();
    let dim = 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_light_axis_1_strided_trailing_batch() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [3, 5, 2].to_vec();
    let dim = 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_light_axis_0_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [5, 2].to_vec();
    let dim = 0;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_light_axis_last_n16() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [1, 9].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
fn irfft_virtual_padding_axis_1_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_padded(client, vec![2, 3, 3], 1, 8);
}

#[test]
fn irfft_virtual_padding_dc_only_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_padded(client, vec![2, 1, 3], 1, 8);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_last_axis() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [5, 2, 1025].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_axis_0() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [33, 2, 1024].to_vec();
    let dim = 0;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_axis_1() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [33, 5, 1024].to_vec();
    let dim = 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_4d_axis_2() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [12, 8, 513, 4].to_vec();
    let dim = 2;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_shared_memory_cap_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [1, 2049, 1].to_vec();
    let dim = 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_large_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [1, 4097, 1].to_vec();
    let dim = 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_four_step_axis_1_strided() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [1, 8193, 1].to_vec();
    let dim = 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_batched_large_axis_last() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [3, 4097].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_large_virtual_padding_matches_materialized_zero_padding() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    test_launch_padded(client, vec![1, 3000], 1, 8192);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_3d_batch_singleton_dim() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [22, 1, 1025].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}

#[test]
#[cfg(feature = "heavy")]
fn irfft_dispatch_more_than_wgpu_x_axis_limit() {
    let client = <TestRuntime as Runtime>::client(&Default::default());
    let spectrum_shape = [65_536, 2].to_vec();
    let dim = spectrum_shape.len() - 1;
    test_launch(client, spectrum_shape, dim);
}
