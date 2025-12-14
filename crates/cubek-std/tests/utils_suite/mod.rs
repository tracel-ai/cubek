use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, TestRuntime};
use cubek_std::test_utils::{
    Distribution, HostData, HostDataType, StrideSpec, TestInput, assert_equals_approx,
};

mod col_major;

#[test]
fn random_uniform_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, expected) = TestInput::random(
        client.clone(),
        vec![4, 4],
        f32::as_type_native_unchecked(),
        42,
        Distribution::Uniform(-1., 1.),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn random_uniform_handle_col_major_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 4];

    let (handle, expected) = TestInput::random(
        client.clone(),
        shape.clone(),
        f32::as_type_native_unchecked(),
        42,
        Distribution::Uniform(-1., 1.),
        StrideSpec::ColMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn random_bernoulli_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, expected) = TestInput::random(
        client.clone(),
        vec![4, 4],
        f32::as_type_native_unchecked(),
        42,
        Distribution::Bernoulli(0.4),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn zeros_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, expected) =
        TestInput::zeros(client.clone(), vec![4, 4], f32::as_type_native_unchecked())
            .generate_with_f32_host_data()
            .unwrap();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn eye_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, expected) =
        TestInput::eye(client.clone(), vec![4, 4], f32::as_type_native_unchecked())
            .generate_with_f32_host_data()
            .unwrap();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn arange_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, expected) = TestInput::arange(
        client.clone(),
        vec![4, 4],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn arange_handle_col_major_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let (handle, expected) = TestInput::arange(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
    )
    .generate_with_f32_host_data()
    .unwrap();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}
