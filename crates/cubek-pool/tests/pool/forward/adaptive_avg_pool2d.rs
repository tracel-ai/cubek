use super::{make_problem, run_pool_test};
use crate::pool::{build_output_tensor, output_host_f32};
use cubecl::{
    ir::{ElemType, FloatKind},
    zspace::Shape,
};
use cubek_pool::{definition::AdaptiveAvgPoolOptions, pool2d};
use cubek_test_utils::TestInput;

const ADAPTIVE_AVG_POOL_TOLERANCE: f32 = 1e-5;

#[test]
fn test_adaptive_avg_pool2d_global() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        Shape::from([2, 7, 7, 512]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [1, 1],
        },
    );
    run_pool_test(
        client,
        1111,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_square_downsample() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        Shape::from([1, 8, 8, 4]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [4, 4],
        },
    );
    run_pool_test(
        client,
        2222,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_non_square() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        Shape::from([1, 10, 6, 8]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [3, 2],
        },
    );
    run_pool_test(
        client,
        3333,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_uneven_indices() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        Shape::from([1, 13, 13, 1]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [3, 3],
        },
    );
    run_pool_test(
        client,
        4444,
        0.0,
        10.0,
        problem,
        ADAPTIVE_AVG_POOL_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_identity() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        Shape::from([2, 4, 4, 16]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [4, 4],
        },
    );
    run_pool_test(
        client,
        5555,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_upsample_logic() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        Shape::from([1, 2, 2, 4]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [4, 4],
        },
    );
    run_pool_test(
        client,
        6666,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_TOLERANCE,
    );
}

/// A window of 4096 ones averages to one. An f16 accumulator stops moving at 2048, where its
/// step size passes the 1.0 being added, so it reaches 2048 and reports 0.5.
#[test]
fn test_adaptive_avg_pool2d_f16_large_global_accumulates_in_f32() {
    let client = cubecl::test_device().client();
    let dtype = ElemType::Float(FloatKind::F16);
    let input_shape = vec![1, 64, 64, 1];
    let input = TestInput::builder(client.clone(), input_shape.clone())
        .dtype(dtype)
        .custom(vec![1.0; input_shape.iter().product()])
        .generate_without_host_data();
    let output = build_output_tensor(&client, vec![1, 1, 1, 1], dtype);
    pool2d(
        &client,
        input.binding(),
        output.clone().binding(),
        AdaptiveAvgPoolOptions::new([1, 1]).into(),
        dtype,
    )
    .expect("f16 adaptive average pool 2d should launch");

    let actual = output_host_f32(&client, output).get_f32(&[0, 0, 0, 0]);
    assert!(actual.is_finite());
    assert!((actual - 1.0).abs() <= 1e-3, "expected 1, got {actual}");
}
