use cubecl::prelude::*;
use cubecl::{TestRuntime, client::ComputeClient, ir::StorageType, std::tensor::TensorHandle};
use cubek_resample::{definition::Resample, resample};
use cubek_test_utils::{HostData, HostDataType, TestInput, assert_equals_approx};

pub fn build_output_tensor(
    client: &ComputeClient<TestRuntime>,
    output_shape: Vec<usize>,
    dtype: StorageType,
) -> TensorHandle<TestRuntime> {
    TestInput::builder(client.clone(), output_shape)
        .dtype(dtype)
        .zeros()
        .generate_without_host_data()
}

pub fn output_host_f32(
    client: &ComputeClient<TestRuntime>,
    output: TensorHandle<TestRuntime>,
) -> HostData {
    HostData::from_tensor_handle(client, output, HostDataType::F32)
}

pub fn validate_test(
    actual: cubek_test_utils::HostData,
    expected: cubek_test_utils::HostData,
    tolerance: f32,
) {
    assert_equals_approx(&actual, &expected, tolerance)
        .as_test_outcome()
        .enforce();
}

pub fn run_test(
    client: &ComputeClient<TestRuntime>,
    input_shape: Vec<usize>,
    input_data: Vec<f32>,
    output_shape: Vec<usize>,
    expected_data: Vec<f32>,
    config: Resample,
) {
    let input_handle = TestInput::builder(client.clone(), input_shape)
        .dtype(f32::as_type_native_unchecked().storage_type())
        .custom(input_data)
        .generate_without_host_data();
    let input = input_handle.clone().binding();

    let output_handle = build_output_tensor(
        &client,
        output_shape.clone(),
        f32::as_type_native_unchecked().storage_type(),
    );
    let output = output_handle.clone().binding();

    resample(
        &client,
        input,
        output,
        config,
        f32::as_type_native_unchecked().storage_type(),
    );

    let actual = output_host_f32(&client, output_handle);
    let expected_handle = TestInput::builder(client.clone(), output_shape)
        .dtype(f32::as_type_native_unchecked().storage_type())
        .custom(expected_data)
        .generate_without_host_data();
    let expected = output_host_f32(&client, expected_handle);

    validate_test(actual, expected, 1e-6);
}
