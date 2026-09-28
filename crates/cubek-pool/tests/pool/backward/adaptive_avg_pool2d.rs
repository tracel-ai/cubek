use super::{make_problem, run_pool_backward_test};
use crate::pool::{build_output_tensor, output_host_f32};
use cubecl::{
    ir::{ElemType, FloatKind},
    zspace::Shape,
};
use cubek_pool::{definition::AdaptiveAvgPoolOptions, pool2d_backward};
use cubek_test_utils::TestInput;

const ADAPTIVE_AVG_POOL_BACKWARD_TOLERANCE: f32 = 1e-5;

#[test]
fn test_adaptive_avg_pool2d_backward_global() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [8, 8],
        Shape::from([2, 1, 1, 4]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [1, 1],
        },
    );
    run_pool_backward_test(
        client,
        7890,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_backward_square() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [7, 7],
        Shape::from([1, 3, 3, 2]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [3, 3],
        },
    );
    run_pool_backward_test(
        client,
        1357,
        -5.0,
        5.0,
        problem,
        ADAPTIVE_AVG_POOL_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_backward_non_square() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [10, 10],
        Shape::from([2, 3, 5, 3]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [3, 5],
        },
    );
    run_pool_backward_test(
        client,
        2468,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_adaptive_avg_pool2d_backward_large_input() {
    let client = cubecl::test_device().client();
    let problem = make_problem(
        [14, 14],
        Shape::from([1, 7, 7, 8]),
        false,
        AdaptiveAvgPoolOptions {
            output_size: [7, 7],
        },
    );
    run_pool_backward_test(
        client,
        9753,
        -1.0,
        1.0,
        problem,
        ADAPTIVE_AVG_POOL_BACKWARD_TOLERANCE,
    );
}

/// Upsampling covers each input pixel with every output cell, so a 1x1 input pooled to 64x64
/// gathers 4096 unit gradients. An f16 accumulator stops moving at 2048 and reports half.
#[test]
fn test_adaptive_avg_pool2d_backward_f16_large_upsample_accumulates_in_f32() {
    let client = cubecl::test_device().client();
    let dtype = ElemType::Float(FloatKind::F16);
    let input_shape = vec![1, 1, 1, 1];
    let out_grad_shape = vec![1, 64, 64, 1];
    let input = build_output_tensor(&client, input_shape.clone(), dtype);
    let out_grad = TestInput::builder(client.clone(), out_grad_shape.clone())
        .dtype(dtype)
        .custom(vec![1.0; out_grad_shape.iter().product()])
        .generate_without_host_data();
    let in_grad = build_output_tensor(&client, input_shape, dtype);

    pool2d_backward(
        &client,
        input.binding(),
        out_grad.binding(),
        in_grad.clone().binding(),
        AdaptiveAvgPoolOptions::new([64, 64]).into(),
        dtype,
    )
    .expect("f16 adaptive average pool 2d backward should launch");

    let actual = output_host_f32(&client, in_grad).get_f32(&[0, 0, 0, 0]);
    assert!(
        (actual - 4096.0).abs() <= 1.0,
        "expected 4096, got {actual}"
    );
}
