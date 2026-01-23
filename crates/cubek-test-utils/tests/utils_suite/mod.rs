use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, TestRuntime};
use cubek_test_utils::{
    DataKind, HostData, HostDataType, StrideSpec, TestInput, assert_equals_approx,
    assert_equals_approx_in_slice,
};

#[test]
fn eye_handle_row_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::new(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Eye,
    )
    .generate();

    let expected = TestInput::new(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: [1., 0., 0., 0., 1., 0.].to_vec(),
        },
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn eye_handle_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::new(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
        DataKind::Eye,
    )
    .generate();

    let expected = TestInput::new(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: [1., 0., 0., 0., 1., 0.].to_vec(),
        },
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_row_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::new(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Arange,
    )
    .generate();

    let expected = TestInput::new(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: [0., 1., 2., 3., 4., 5.].to_vec(),
        },
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    let handle = TestInput::new(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
        DataKind::Arange,
    )
    .generate();

    let expected = TestInput::new(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: [0., 1., 2., 3., 4., 5.].to_vec(),
        },
    )
    .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, &handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn custom_handle_row_major_col_major() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let contiguous_data = [9., 8., 7., 6., 5., 4.].to_vec();

    let (_, row_major) = TestInput::new(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: contiguous_data.clone(),
        },
    )
    .generate_with_f32_host_data();

    let (_, col_major) = TestInput::new(
        client.clone(),
        vec![2, 3],
        f32::as_type_native_unchecked(),
        StrideSpec::ColMajor,
        DataKind::Custom {
            data: contiguous_data,
        },
    )
    .generate_with_f32_host_data();

    assert_equals_approx(&col_major, &row_major, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_row_major_slice() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![2, 3];

    // Create an "actual" tensor where the second row differs
    let actual_data = [0., 1., 2., 9., 9., 9.].to_vec(); // last 3 elements differ
    let actual = TestInput::new(
        client.clone(),
        shape.clone(),
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Custom { data: actual_data },
    )
    .f32_host_data();

    // Expected tensor
    let expected_data = [0., 1., 2., 3., 4., 5.].to_vec();
    let expected = TestInput::new(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        StrideSpec::RowMajor,
        DataKind::Custom {
            data: expected_data,
        },
    )
    .f32_host_data();

    assert_equals_approx_in_slice(&actual, &expected, 0.001, vec![0..1, 0..3])
        .as_test_outcome()
        .enforce();
}
