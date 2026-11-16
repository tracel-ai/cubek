use crate::pool::{PoolForward, Pooled, Seeded};
use crate::pool::{build_output_tensor, output_host_f32};
use cubecl::{
    ir::{ElemType, FloatKind},
    prelude::*,
    zspace::Shape,
};
use cubek_pool::{
    definition::{
        AdaptiveAvgPoolOptions, AvgPoolOptions, MaxPoolOptions, PoolError, PoolForwardProblem,
        PoolMode,
    },
    pool3d,
};
use cubek_test_utils::ScopeUnderTest;
use cubek_test_utils::TestInput;

const TOLERANCE: f32 = 1e-5;

#[test]
fn test_adaptive_avg_pool3d_global() {
    PoolForward.check([Pooled::adaptive([2, 9, 11, 7, 8], [1, 1, 1], 1001)]);
}

#[test]
fn test_adaptive_avg_pool3d_divisible_downsample_vectorized_channels() {
    PoolForward.check([Pooled::adaptive([2, 6, 8, 10, 8], [3, 4, 5], 1002)]);
}

#[test]
fn test_adaptive_avg_pool3d_non_divisible_asymmetric_odd_channels() {
    PoolForward.check([Pooled::adaptive([2, 5, 7, 4, 7], [3, 4, 3], 1003)]);
}

#[test]
fn test_adaptive_avg_pool3d_non_divisible_large_windows() {
    PoolForward.check([Pooled::adaptive([1, 19, 20, 15, 8], [2, 3, 2], 1005)]);
}

#[test]
fn test_adaptive_avg_pool3d_output_larger_than_input() {
    PoolForward.check([Pooled::adaptive([1, 2, 3, 2, 5], [4, 5, 3], 1004)]);
}

#[test]
fn test_adaptive_avg_pool3d_large_dimension_index_math() {
    PoolForward.check([Pooled::adaptive(
        [1, 100_000, 1, 1, 1],
        [100_000, 1, 1],
        1006,
    )]);
}

#[test]
fn test_adaptive_avg_pool3d_validates_rank_batch_channel_and_output_size() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let options = AdaptiveAvgPoolOptions::new([3, 4, 3]);

    let invalid_rank_input = build_output_tensor(&client, vec![2, 5, 7, 3], dtype);
    let valid_output = build_output_tensor(&client, vec![2, 3, 4, 3, 3], dtype);
    assert!(matches!(
        pool3d(
            &client,
            invalid_rank_input.binding(),
            valid_output.clone().binding(),
            options.clone().into(),
            dtype,
        ),
        Err(PoolError::InvalidRank { .. })
    ));

    let input = build_output_tensor(&client, vec![2, 5, 7, 4, 3], dtype);
    let batch_mismatch = build_output_tensor(&client, vec![1, 3, 4, 3, 3], dtype);
    assert!(matches!(
        pool3d(
            &client,
            input.clone().binding(),
            batch_mismatch.binding(),
            options.clone().into(),
            dtype,
        ),
        Err(PoolError::BatchMismatch { .. })
    ));

    let channel_mismatch = build_output_tensor(&client, vec![2, 3, 4, 3, 4], dtype);
    assert!(matches!(
        pool3d(
            &client,
            input.clone().binding(),
            channel_mismatch.binding(),
            options.clone().into(),
            dtype,
        ),
        Err(PoolError::ChannelMismatch { .. })
    ));

    let output_size_mismatch = build_output_tensor(&client, vec![2, 3, 4, 2, 3], dtype);
    assert!(matches!(
        pool3d(
            &client,
            input.binding(),
            output_size_mismatch.binding(),
            options.into(),
            dtype,
        ),
        Err(PoolError::OutputSizeMismatch { .. })
    ));
}

#[test]
fn test_adaptive_avg_pool3d_rejects_zero_spatial_dimensions() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    let input = build_output_tensor(&client, vec![1, 0, 2, 2, 1], dtype);
    let output = build_output_tensor(&client, vec![1, 1, 1, 1, 1], dtype);
    assert!(matches!(
        pool3d(
            &client,
            input.binding(),
            output.binding(),
            AdaptiveAvgPoolOptions::new([1, 1, 1]).into(),
            dtype,
        ),
        Err(PoolError::InvalidSpatialSize { .. })
    ));

    let input = build_output_tensor(&client, vec![1, 2, 2, 2, 1], dtype);
    let output = build_output_tensor(&client, vec![1, 0, 1, 1, 1], dtype);
    assert!(matches!(
        pool3d(
            &client,
            input.binding(),
            output.binding(),
            AdaptiveAvgPoolOptions::new([0, 1, 1]).into(),
            dtype,
        ),
        Err(PoolError::InvalidSpatialSize { .. })
    ));
}

#[test]
fn test_adaptive_avg_pool3d_accepts_empty_batch_and_channels() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();

    for (input_shape, output_shape) in [
        (vec![0, 2, 2, 2, 1], vec![0, 1, 1, 1, 1]),
        (vec![1, 2, 2, 2, 0], vec![1, 1, 1, 1, 0]),
    ] {
        let input = build_output_tensor(&client, input_shape, dtype);
        let output = build_output_tensor(&client, output_shape, dtype);
        pool3d(
            &client,
            input.binding(),
            output.binding(),
            AdaptiveAvgPoolOptions::new([1, 1, 1]).into(),
            dtype,
        )
        .expect("empty batch and channel dimensions should produce empty output");
    }
}

#[test]
fn test_adaptive_avg_pool3d_rejects_other_pool_modes() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let input = build_output_tensor(&client, vec![1, 4, 4, 4, 1], dtype);
    let output = build_output_tensor(&client, vec![1, 2, 2, 2, 1], dtype);

    for mode in [
        PoolMode::Avg(AvgPoolOptions::new(
            [2, 2, 2],
            [2, 2, 2],
            [0, 0, 0],
            false,
            false,
        )),
        PoolMode::Max(MaxPoolOptions::new(
            [2, 2, 2],
            [2, 2, 2],
            [0, 0, 0],
            [1, 1, 1],
            false,
        )),
    ] {
        assert!(matches!(
            pool3d(
                &client,
                input.clone().binding(),
                output.clone().binding(),
                mode,
                dtype,
            ),
            Err(PoolError::UnsupportedMode { .. })
        ));
    }
}

#[test]
fn test_adaptive_avg_pool3d_f16_large_global_accumulates_in_f32() {
    let client = cubecl::test_device().client();
    let dtype = ElemType::Float(FloatKind::F16);
    let input_shape = vec![1, 41, 41, 41, 1];
    let input = TestInput::builder(client.clone(), input_shape.clone())
        .dtype(dtype)
        .custom(vec![1.0; input_shape.iter().product()])
        .generate_without_host_data();
    let output = build_output_tensor(&client, vec![1, 1, 1, 1, 1], dtype);
    pool3d(
        &client,
        input.binding(),
        output.clone().binding(),
        AdaptiveAvgPoolOptions::new([1, 1, 1]).into(),
        dtype,
    )
    .expect("f16 adaptive average pool 3d should launch");

    let actual = output_host_f32(&client, output).get_f32(&[0, 0, 0, 0, 0]);
    assert!(actual.is_finite());
    assert!((actual - 1.0).abs() <= 1e-3, "expected 1, got {actual}");
}

impl Pooled<3> {
    /// `input_shape` pooled to `output_size` by an adaptive average, its input seeded by `seed` in
    /// `[-1, 1)`.
    fn adaptive(input_shape: [usize; 5], output_size: [usize; 3], seed: u64) -> Self {
        Self {
            problem: PoolForwardProblem {
                input_shape: Shape::from(input_shape),
                with_indices: false,
                mode: AdaptiveAvgPoolOptions::new(output_size).into(),
            },
            input: Seeded {
                seed,
                inputs: -1.0..1.0,
            },
            tolerance: TOLERANCE,
        }
    }
}
