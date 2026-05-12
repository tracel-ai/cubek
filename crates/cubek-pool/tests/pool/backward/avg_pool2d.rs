use super::{make_problem, run_pool_backward_test};
use cubecl::{Runtime, TestRuntime, zspace::Shape};
use cubek_pool::definition::AvgPoolOptions;

const AVG_POOL2D_BACKWARD_TOLERANCE: f32 = 0.000001;

#[test]
fn test_avg_pool2d_backward() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [4, 4],
        Shape::from([2, 4, 4, 2]),
        false,
        AvgPoolOptions::new([3, 3], [1, 1], [1, 1], false, false),
    );
    run_pool_backward_test(
        client,
        5678,
        -10.0,
        10.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_strided_no_pad() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [6, 6],
        Shape::from([2, 3, 3, 4]),
        false,
        AvgPoolOptions::new([2, 2], [2, 2], [0, 0], false, false),
    );
    run_pool_backward_test(
        client,
        1234,
        -1.0,
        1.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_exclude_pad() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 5],
        Shape::from([1, 3, 3, 1]),
        false,
        AvgPoolOptions::new([3, 3], [2, 2], [1, 1], false, false),
    );
    run_pool_backward_test(
        client,
        9999,
        -1.0,
        1.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_non_square_asymmetric() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 7],
        Shape::from([2, 6, 3, 3]),
        false,
        AvgPoolOptions::new([2, 3], [1, 2], [1, 0], false, true),
    );
    run_pool_backward_test(
        client,
        3456,
        -1.0,
        1.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_avg_pool2d_backward_ceil_mode() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [5, 5],
        Shape::from([2, 3, 3, 4]),
        false,
        AvgPoolOptions::new([2, 2], [2, 2], [0, 0], true, true),
    );
    run_pool_backward_test(
        client,
        4567,
        -1.0,
        1.0,
        problem,
        AVG_POOL2D_BACKWARD_TOLERANCE,
    );
}
