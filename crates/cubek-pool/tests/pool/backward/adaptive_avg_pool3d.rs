use crate::pool::{build_output_tensor, output_host_f32, validate_test};
use cubecl::{
    Runtime, TestRuntime,
    ir::{ElemType, FloatKind},
    prelude::*,
    zspace::Shape,
};
use cubek_pool::{
    definition::{
        AdaptiveAvgPoolOptions, AvgPoolOptions, MaxPoolOptions, PoolBackwardProblem, PoolError,
        PoolMode,
    },
    eval::cpu_reference::cpu_reference_pool_backward,
    pool3d_backward,
};
use cubek_test_utils::TestInput;

const TOLERANCE: f32 = 1e-5;

#[test]
fn test_adaptive_avg_pool3d_backward_global() {
    run_case([5, 7, 4], [2, 1, 1, 1, 3], [1, 1, 1], 2001);
}

#[test]
fn test_adaptive_avg_pool3d_backward_divisible_vectorized_channels() {
    run_case([6, 8, 10], [2, 3, 4, 5, 8], [3, 4, 5], 2002);
}

#[test]
fn test_adaptive_avg_pool3d_backward_non_divisible_overlap_odd_channels() {
    run_case([5, 7, 4], [2, 3, 4, 3, 7], [3, 4, 3], 2003);
}

#[test]
fn test_adaptive_avg_pool3d_backward_output_larger_than_input() {
    run_case([2, 3, 2], [1, 4, 5, 3, 5], [4, 5, 3], 2004);
}

#[test]
fn test_adaptive_avg_pool3d_backward_large_dimension_index_math() {
    run_case(
        [100_000, 1, 1],
        [1, 100_000, 1, 1, 1],
        [100_000, 1, 1],
        2005,
    );
}

#[test]
fn test_adaptive_avg_pool3d_backward_conserves_gradient_sum() {
    let client = TestRuntime::client(&Default::default());
    let dtype = f32::elem_type_native();
    let input = build_output_tensor(&client, vec![2, 5, 7, 4, 3], dtype);
    let (out_grad, out_grad_host) = TestInput::builder(client.clone(), [2, 3, 4, 3, 3])
        .uniform(2112, -1.0, 1.0)
        .generate_with_f32_host_data();
    let in_grad = build_output_tensor(&client, vec![2, 5, 7, 4, 3], dtype);

    pool3d_backward(
        &client,
        input.clone().binding(),
        out_grad.binding(),
        in_grad.clone().binding(),
        AdaptiveAvgPoolOptions::new([3, 4, 3]).into(),
        dtype,
    )
    .expect("adaptive average pool 3d backward should launch");

    let in_grad_host = output_host_f32(&client, in_grad);
    let expected_sum: f32 = out_grad_host
        .iter_indexed_f32()
        .map(|(_, value)| value)
        .sum();
    let actual_sum: f32 = in_grad_host
        .iter_indexed_f32()
        .map(|(_, value)| value)
        .sum();
    let tolerance = 1e-4 * expected_sum.abs().max(1.0);
    assert!(
        (actual_sum - expected_sum).abs() <= tolerance,
        "input-gradient sum {actual_sum} differs from output-gradient sum {expected_sum}"
    );
}

#[test]
fn test_adaptive_avg_pool3d_backward_validates_output_and_input_gradient_shapes() {
    let client = TestRuntime::client(&Default::default());
    let dtype = f32::elem_type_native();
    let input = build_output_tensor(&client, vec![2, 5, 7, 4, 3], dtype);
    let options = AdaptiveAvgPoolOptions::new([3, 4, 3]);

    let out_grad_mismatch = build_output_tensor(&client, vec![2, 3, 4, 2, 3], dtype);
    let in_grad = build_output_tensor(&client, vec![2, 5, 7, 4, 3], dtype);
    assert!(matches!(
        pool3d_backward(
            &client,
            input.clone().binding(),
            out_grad_mismatch.binding(),
            in_grad.clone().binding(),
            options.clone().into(),
            dtype,
        ),
        Err(PoolError::OutputSizeMismatch { .. })
    ));

    let out_grad = build_output_tensor(&client, vec![2, 3, 4, 3, 3], dtype);
    let in_grad_mismatch = build_output_tensor(&client, vec![2, 5, 7, 5, 3], dtype);
    assert!(matches!(
        pool3d_backward(
            &client,
            input.binding(),
            out_grad.binding(),
            in_grad_mismatch.binding(),
            options.into(),
            dtype,
        ),
        Err(PoolError::InputGradientShapeMismatch { .. })
    ));
}

#[test]
fn test_adaptive_avg_pool3d_backward_rejects_zero_spatial_dimensions() {
    let client = TestRuntime::client(&Default::default());
    let dtype = f32::elem_type_native();

    let input = build_output_tensor(&client, vec![1, 0, 2, 2, 1], dtype);
    let out_grad = build_output_tensor(&client, vec![1, 1, 1, 1, 1], dtype);
    let in_grad = build_output_tensor(&client, vec![1, 0, 2, 2, 1], dtype);
    assert!(matches!(
        pool3d_backward(
            &client,
            input.binding(),
            out_grad.binding(),
            in_grad.binding(),
            AdaptiveAvgPoolOptions::new([1, 1, 1]).into(),
            dtype,
        ),
        Err(PoolError::InvalidSpatialSize { .. })
    ));

    let input = build_output_tensor(&client, vec![1, 2, 2, 2, 1], dtype);
    let out_grad = build_output_tensor(&client, vec![1, 0, 1, 1, 1], dtype);
    let in_grad = build_output_tensor(&client, vec![1, 2, 2, 2, 1], dtype);
    assert!(matches!(
        pool3d_backward(
            &client,
            input.binding(),
            out_grad.binding(),
            in_grad.binding(),
            AdaptiveAvgPoolOptions::new([0, 1, 1]).into(),
            dtype,
        ),
        Err(PoolError::InvalidSpatialSize { .. })
    ));
}

#[test]
fn test_adaptive_avg_pool3d_backward_accepts_empty_batch_and_channels() {
    let client = TestRuntime::client(&Default::default());
    let dtype = f32::elem_type_native();

    for (input_shape, out_grad_shape) in [
        (vec![0, 2, 2, 2, 1], vec![0, 1, 1, 1, 1]),
        (vec![1, 2, 2, 2, 0], vec![1, 1, 1, 1, 0]),
    ] {
        let input = build_output_tensor(&client, input_shape.clone(), dtype);
        let out_grad = build_output_tensor(&client, out_grad_shape, dtype);
        let in_grad = build_output_tensor(&client, input_shape, dtype);
        pool3d_backward(
            &client,
            input.binding(),
            out_grad.binding(),
            in_grad.binding(),
            AdaptiveAvgPoolOptions::new([1, 1, 1]).into(),
            dtype,
        )
        .expect("empty batch and channel dimensions should produce empty input gradients");
    }
}

#[test]
fn test_adaptive_avg_pool3d_backward_rejects_other_pool_modes() {
    let client = TestRuntime::client(&Default::default());
    let dtype = f32::elem_type_native();
    let input = build_output_tensor(&client, vec![1, 4, 4, 4, 1], dtype);
    let out_grad = build_output_tensor(&client, vec![1, 2, 2, 2, 1], dtype);
    let in_grad = build_output_tensor(&client, vec![1, 4, 4, 4, 1], dtype);

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
            pool3d_backward(
                &client,
                input.clone().binding(),
                out_grad.clone().binding(),
                in_grad.clone().binding(),
                mode,
                dtype,
            ),
            Err(PoolError::UnsupportedMode { .. })
        ));
    }
}

#[test]
fn test_adaptive_avg_pool3d_backward_f16_large_global_accumulates_in_f32() {
    let client = TestRuntime::client(&Default::default());
    let dtype = ElemType::Float(FloatKind::F16);
    let input_shape = vec![1, 41, 41, 41, 1];
    let input = build_output_tensor(&client, input_shape.clone(), dtype);
    let out_grad = TestInput::builder(client.clone(), [1, 1, 1, 1, 1])
        .dtype(dtype)
        .custom(vec![1.0])
        .generate_without_host_data();
    let in_grad = build_output_tensor(&client, input_shape, dtype);

    pool3d_backward(
        &client,
        input.binding(),
        out_grad.binding(),
        in_grad.clone().binding(),
        AdaptiveAvgPoolOptions::new([1, 1, 1]).into(),
        dtype,
    )
    .expect("f16 adaptive average pool 3d backward should launch");

    let actual = output_host_f32(&client, in_grad);
    let mut sum = 0.0;
    for (_, value) in actual.iter_indexed_f32() {
        assert!(value.is_finite());
        assert!(value > 0.0, "expected a non-zero input gradient");
        sum += value;
    }
    assert!((sum - 1.0).abs() <= 1e-2, "expected sum 1, got {sum}");
}

fn run_case(
    input_size: [usize; 3],
    out_grad_shape: [usize; 5],
    output_size: [usize; 3],
    seed: u64,
) {
    let client = TestRuntime::client(&Default::default());
    let problem = PoolBackwardProblem {
        input_size,
        out_grad_shape: Shape::from(out_grad_shape),
        with_indices: false,
        mode: AdaptiveAvgPoolOptions::new(output_size).into(),
    };
    let input_shape = vec![
        problem.out_grad_shape[0],
        problem.input_size[0],
        problem.input_size[1],
        problem.input_size[2],
        problem.out_grad_shape[4],
    ];
    let (input, input_data) = TestInput::builder(client.clone(), input_shape.clone())
        .uniform(seed, -1.0, 1.0)
        .generate_with_f32_host_data();
    let (out_grad, out_grad_data) =
        TestInput::builder(client.clone(), problem.out_grad_shape.to_vec())
            .uniform(seed + 1, -1.0, 1.0)
            .generate_with_f32_host_data();
    let in_grad = build_output_tensor(&client, input_shape, input.dtype);

    let result = pool3d_backward(
        &client,
        input.clone().binding(),
        out_grad.binding(),
        in_grad.clone().binding(),
        problem.mode.clone(),
        input.dtype,
    );
    let actual = output_host_f32(&client, in_grad);
    let expected = cpu_reference_pool_backward(&out_grad_data, &input_data, problem);

    validate_test(result, actual, expected, TOLERANCE);
}
