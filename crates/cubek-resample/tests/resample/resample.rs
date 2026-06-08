use crate::resample::run_test;
use cubecl::TestRuntime;
use cubecl::prelude::*;

#[test]
fn resample_1d_simple_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![4];
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];
    let scales = vec![1.0f32];

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        scales,
    );
}

#[test]
fn resample_1d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![8];
    let expected_data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    let scales = vec![0.5f32];

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        scales,
    );
}

#[test]
fn resample_nhwc_2d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![1, 2, 2, 1];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![1, 4, 4, 1];
    let expected_data = vec![
        1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 3.0, 3.0, 4.0, 4.0,
    ];
    let scales = vec![1.0f32, 0.5, 0.5, 1.0];

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        scales,
    );
}
