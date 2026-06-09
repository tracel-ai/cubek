use crate::resample::run_test;
use cubecl::{Runtime, TestRuntime};
use cubek_resample::definition::{Kernel, Placement, Resample, Semiring};

#[test]
fn resample_1d_identity_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let output_shape = vec![4];
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];

    let config = Resample::new()
        .with_kernel(Kernel::One)
        .with_placement(Placement::Continuous {
            scale: 1.0,
            offset: 0.0,
        })
        .with_semiring(Semiring::Linear)
        .with_axis(0);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        config,
    );
}

#[test]
fn resample_1d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let output_shape = vec![8];
    let expected_data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];

    let config = Resample::new()
        .with_kernel(Kernel::One)
        .with_placement(Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        })
        .with_semiring(Semiring::Linear)
        .with_axis(0);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        config,
    );
}

#[test]
fn resample_nhwc_2d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![1, 2, 2, 1];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let output_shape = vec![1, 4, 2, 1];
    let expected_output = vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0];

    let config = Resample::new()
        .with_kernel(Kernel::One)
        .with_placement(Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        })
        .with_semiring(Semiring::Linear)
        .with_axis(0)
        .with_axis(1);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_output,
        config,
    );
}

#[test]
fn resample_nhwc_3d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![2, 2, 2];
    let input_data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];

    let output_shape = vec![1, 2, 1];
    let expected_output = vec![1.0, 3.0];

    let config = Resample::new()
        .with_kernel(Kernel::One)
        .with_placement(Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        })
        .with_semiring(Semiring::Linear)
        .with_axis(0)
        .with_axis(2);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_output,
        config,
    );
}

#[test]
fn resample_nhwc_2d_upsample_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![1, 2, 2, 1];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];

    let output_shape = vec![1, 4, 2, 4];
    let expected_output = vec![
        1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 2.0, 2.0, 3.0, 3.0, 3.0, 3.0, 4.0, 4.0, 4.0,
        4.0, // This test is wrong.
    ];

    let config = Resample::new()
        .with_kernel(Kernel::One)
        .with_placement(Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        })
        .with_semiring(Semiring::Linear)
        .with_axis(0)
        .with_axis(2);

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_output,
        config,
    );
}
