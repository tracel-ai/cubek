use super::{make_problem, run_pool_backward_test};
use cubecl::{Runtime, TestRuntime, zspace::Shape};
use cubek_pool::definition::AdaptiveAvgPoolOptions;

const ADAPTIVE_AVG_POOL_BACKWARD_TOLERANCE: f32 = 1e-5;

#[test]
fn test_adaptive_avg_pool2d_backward_global() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [8, 8],
        Shape::from([2, 4, 1, 1]),
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
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [7, 7],
        Shape::from([1, 2, 3, 3]),
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
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [10, 10],
        Shape::from([2, 3, 3, 5]),
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
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [14, 14],
        Shape::from([1, 8, 7, 7]),
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
