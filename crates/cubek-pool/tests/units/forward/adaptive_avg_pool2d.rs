use super::{make_problem, run_pool_test};
use cubecl::{Runtime, TestRuntime, zspace::Shape};
use cubek_pool::definition::AdaptiveAvgPoolOptions;

const ADAPTIVE_AVG_POOL_TOLERANCE: f32 = 1e-5;

#[test]
fn test_adaptive_avg_pool2d_global() {
    let client = TestRuntime::client(&Default::default());
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
    let client = TestRuntime::client(&Default::default());
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
    let client = TestRuntime::client(&Default::default());
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
    let client = TestRuntime::client(&Default::default());
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
    let client = TestRuntime::client(&Default::default());
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
    let client = TestRuntime::client(&Default::default());
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
