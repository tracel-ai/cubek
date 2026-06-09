use crate::resample::run_test;
use cubecl::{Runtime, TestRuntime, prelude::Sequence};
use cubek_resample::definition::{Kernel, Placement, Resample, Semiring};

/// Nearest-neighbor 1x identity
#[test]
fn resample_1d_identity_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![4];
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];

    let mut reduce_axes = Sequence::new();
    reduce_axes.push(0);

    let config = Resample {
        kernel: Kernel::One,
        placement: Placement::Continuous {
            scale: 1.0,
            offset: 0.0,
        },
        semiring: Semiring::Linear,
        reduce_axes,
    };

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        config,
    );
}

/// Nearest-neighbor 2× upscale (Delta + Affine{scale=2} + Linear).
#[test]
fn resample_1d_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![8];
    let expected_data = vec![1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0];

    let mut reduce_axes = Sequence::new();
    reduce_axes.push(0);

    let config = Resample {
        kernel: Kernel::One,
        placement: Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        },
        semiring: Semiring::Linear,
        reduce_axes,
    };

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        config,
    );
}

/// Nearest-neighbor 2× upscale on NHWC 2D (separable: H then W).
/// For now we test the single-axis version on axis 1 (H dimension).
#[test]
fn resample_nhwc_2d_test() {
    let client = TestRuntime::client(&Default::default());

    // Input: [1, 2, 2, 1] NHWC
    let input_shape = vec![1, 2, 2, 1];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    // Upscale H: [1, 2, 2, 1] → [1, 4, 2, 1]
    let intermediate_shape = vec![1, 4, 2, 1];
    let expected_intermediate = vec![1.0, 2.0, 1.0, 2.0, 3.0, 4.0, 3.0, 4.0];

    let mut reduce_axes = Sequence::new();
    reduce_axes.push(0);
    reduce_axes.push(1);

    let config = Resample {
        kernel: Kernel::One,
        placement: Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        },
        semiring: Semiring::Linear,
        reduce_axes,
    };

    run_test(
        &client,
        input_shape,
        input_data,
        intermediate_shape,
        expected_intermediate,
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

    let mut reduce_axes = Sequence::new();
    reduce_axes.push(0);
    reduce_axes.push(2);

    let config = Resample {
        kernel: Kernel::One,
        placement: Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        },
        semiring: Semiring::Linear,
        reduce_axes,
    };

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_output,
        config,
    );
}
