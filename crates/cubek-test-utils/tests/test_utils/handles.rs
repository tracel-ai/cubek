use cubecl::zspace::shape;
use cubek_test_utils::{
    HostData, HostDataType, StridedLayout, TestInput, assert_equals_approx,
    assert_equals_approx_in_slice,
};

#[test]
fn eye_handle_row_major() {
    let client = cubecl::test_device().client();

    let shape = [2, 3];

    let handle = TestInput::builder(client.clone(), shape)
        .eye()
        .generate_without_host_data();

    let expected = TestInput::builder(client.clone(), [2, 3])
        .custom(vec![1., 0., 0., 0., 1., 0.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn eye_handle_col_major() {
    let client = cubecl::test_device().client();

    let shape = [2, 3];

    let handle = TestInput::builder(client.clone(), shape)
        .layout(StridedLayout::ColMajor)
        .eye()
        .generate_without_host_data();

    let expected = TestInput::builder(client.clone(), [2, 3])
        .custom(vec![1., 0., 0., 0., 1., 0.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_row_major() {
    let client = cubecl::test_device().client();

    let shape = shape![2, 3];

    let handle = TestInput::builder(client.clone(), shape)
        .arange()
        .generate_without_host_data();

    let expected = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 3., 4., 5.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_col_major() {
    let client = cubecl::test_device().client();

    let shape = shape![2, 3];

    let handle = TestInput::builder(client.clone(), shape)
        .layout(StridedLayout::ColMajor)
        .arange()
        .generate_without_host_data();

    let expected = TestInput::builder(client.clone(), shape![2, 3])
        .custom(vec![0., 1., 2., 3., 4., 5.])
        .f32_host_data();

    let actual = HostData::from_tensor_handle(&client, handle, HostDataType::F32);

    assert_equals_approx(&actual, &expected, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn custom_handle_row_major_col_major() {
    let client = cubecl::test_device().client();

    let contiguous_data = [9., 8., 7., 6., 5., 4.].to_vec();

    let (_, row_major) = TestInput::builder(client.clone(), shape![2, 3])
        .custom(contiguous_data.clone())
        .generate_with_f32_host_data();

    let (_, col_major) = TestInput::builder(client.clone(), shape![2, 3])
        .layout(StridedLayout::ColMajor)
        .custom(contiguous_data)
        .generate_with_f32_host_data();

    assert_equals_approx(&col_major, &row_major, 0.001)
        .as_test_outcome()
        .enforce();
}

#[test]
fn arange_handle_row_major_slice() {
    let client = cubecl::test_device().client();

    let shape = shape![2, 3];

    // Create an "actual" tensor where the second row differs
    let actual_data = vec![0., 1., 2., 9., 9., 9.]; // last 3 elements differ
    let actual = TestInput::builder(client.clone(), shape.clone())
        .custom(actual_data)
        .f32_host_data();

    // Expected tensor
    let expected_data = vec![0., 1., 2., 3., 4., 5.];
    let expected = TestInput::builder(client.clone(), shape)
        .custom(expected_data)
        .f32_host_data();

    assert_equals_approx_in_slice(&actual, &expected, 0.001, vec![0..1, 0..3])
        .as_test_outcome()
        .enforce();
}
