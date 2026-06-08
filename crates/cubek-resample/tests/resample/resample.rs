use crate::resample::run_test;
use cubecl::TestRuntime;
use cubecl::prelude::*;
use cubek_resample::definition::GlobalOperation;
use cubek_resample::definition::Semiring;

#[test]
fn resample_1d_simple_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![4];
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];
    let global_operation = GlobalOperation::Scalar(Semiring::Sum);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        global_operation,
    );
}

#[test]
fn resample_1d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![8];
    let expected_data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];
    let global_operation = GlobalOperation::Scalar(Semiring::Sum);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        global_operation,
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
    let global_operation = GlobalOperation::Scalar(Semiring::Sum);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        global_operation,
    );
}
