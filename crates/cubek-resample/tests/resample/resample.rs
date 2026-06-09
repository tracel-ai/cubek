use crate::resample::run_test;
use cubecl::{Runtime, TestRuntime};
use cubek_resample::definition::{Kernel, Placement, Resample, Semiring};

/// Nearest-neighbor 1x identity
#[test]
fn resample_1d_identity_test() {
    let client = TestRuntime::client(&Default::default());

    let input_shape = vec![4];
    let input_data = vec![1.0, 2.0, 3.0, 4.0];
    let output_shape = vec![4];
    let expected_data = vec![1.0, 2.0, 3.0, 4.0];

    let config = Resample {
        kernel: Kernel::One,
        placement: Placement::Continuous {
            scale: 1.0,
            offset: 0.0,
        },
        semiring: Semiring::Linear,
    };

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        config,
        0,
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

    let config = Resample {
        kernel: Kernel::One,
        placement: Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        },
        semiring: Semiring::Linear,
    };

    run_test(
        &client,
        input_shape,
        input_data,
        output_shape,
        expected_data,
        config,
        0,
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

    let config = Resample {
        kernel: Kernel::One,
        placement: Placement::Continuous {
            scale: 0.5,
            offset: 0.0,
        },
        semiring: Semiring::Linear,
    };

    run_test(
        &client,
        input_shape,
        input_data,
        intermediate_shape,
        expected_intermediate,
        config,
        1, // spatial axis = H (axis 1 in NHWC)
    );
}
