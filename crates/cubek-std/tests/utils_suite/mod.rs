use cubecl::frontend::CubePrimitive;
use cubecl::{Runtime, TestRuntime};
use cubek_std::test_utils::{
    Distribution, TestInput, assert_equals_approx, batched_matrix_strides,
};

#[test]
fn random_uniform_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, host_data) = TestInput::random(
        client.clone(),
        vec![4, 4],
        f32::as_type_native_unchecked(),
        42,
        Distribution::Uniform(-1., 1.),
        None,
    )
    .generate_with_f32_host_data()
    .unwrap();

    assert_equals_approx(&client, &handle, &host_data.into_f32(), 0.001).unwrap();
}

#[test]
fn random_uniform_handle_col_major_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![4, 4];
    let strides = batched_matrix_strides(&shape, true);

    let (handle, host_data) = TestInput::random(
        client.clone(),
        shape.clone(),
        f32::as_type_native_unchecked(),
        42,
        Distribution::Uniform(-1., 1.),
        Some(strides),
    )
    .generate_with_f32_host_data()
    .unwrap();

    // handle.strides = batched_matrix_strides(&shape, false);
    // println!("{:?}", handle);
    assert_equals_approx(&client, &handle, &host_data.into_f32(), 0.001).unwrap();
}

#[test]
fn random_bernoulli_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, host_data) = TestInput::random(
        client.clone(),
        vec![4, 4],
        f32::as_type_native_unchecked(),
        42,
        Distribution::Bernoulli(0.4),
        None,
    )
    .generate_with_f32_host_data()
    .unwrap();

    assert_equals_approx(&client, &handle, &host_data.into_f32(), 0.001).unwrap();
}

#[test]
fn zeros_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, host_data) =
        TestInput::zeros(client.clone(), vec![4, 4], f32::as_type_native_unchecked())
            .generate_with_f32_host_data()
            .unwrap();

    assert_equals_approx(&client, &handle, &host_data.into_f32(), 0.001).unwrap();
}

#[test]
fn eye_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, host_data) =
        TestInput::eye(client.clone(), vec![4, 4], f32::as_type_native_unchecked())
            .generate_with_f32_host_data()
            .unwrap();

    assert_equals_approx(&client, &handle, &host_data.into_f32(), 0.001).unwrap();
}

#[test]
fn arange_handle_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let (handle, host_data) = TestInput::arange(
        client.clone(),
        vec![4, 4],
        f32::as_type_native_unchecked(),
        None,
    )
    .generate_with_f32_host_data()
    .unwrap();

    assert_equals_approx(&client, &handle, &host_data.into_f32(), 0.001).unwrap();
}

#[test]
fn arange_handle_col_major_equal_to_host_data() {
    let client = <TestRuntime as Runtime>::client(&Default::default());

    let shape = vec![4, 4];
    let strides = batched_matrix_strides(&shape, true);

    let (handle, host_data) = TestInput::arange(
        client.clone(),
        shape,
        f32::as_type_native_unchecked(),
        Some(strides),
    )
    .generate_with_f32_host_data()
    .unwrap();

    assert_equals_approx(&client, &handle, &host_data.into_f32(), 0.001).unwrap();
}
