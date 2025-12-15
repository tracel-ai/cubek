use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, TestRuntime};
use cubek_std::test_utils::{HostData, HostDataType, StrideSpec, TestInput, assert_equals_approx};

#[test]
fn eye_handle_row_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::eye(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
    )
    .generate();

    let expected = TestInput::custom(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        [1., 0., 0., 0., 1., 0.].to_vec(),
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn eye_handle_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::eye(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
    )
    .generate();

    let expected = TestInput::custom(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        [1., 0., 0., 0., 1., 0.].to_vec(),
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn arange_handle_row_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::arange(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
    )
    .generate();

    let expected = TestInput::custom(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        [0., 1., 2., 3., 4., 5.].to_vec(),
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn arange_handle_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::arange(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
    )
    .generate();

    let expected = TestInput::custom(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        [0., 1., 2., 3., 4., 5.].to_vec(),
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001).unwrap();
}

#[test]
fn custom_handle_row_major_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let contiguous_data = [9., 8., 7., 6., 5., 4.].to_vec();

    let (_, row_major) = TestInput::custom(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        contiguous_data.clone(),
    )
    .generate_with_f32_host_data();

    let (_, col_major) = TestInput::custom(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
        contiguous_data,
    )
    .generate_with_f32_host_data();

    assert_equals_approx(&col_major, &row_major, 0.001).unwrap();
}
