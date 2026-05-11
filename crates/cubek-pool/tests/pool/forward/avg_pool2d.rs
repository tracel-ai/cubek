use super::{make_problem, run_pool_test};
use cubecl::{Runtime, TestRuntime, zspace::Shape};
use cubek_pool::definition::AvgPoolOptions;

const AVG_POOL2D_TOLERANCE: f32 = 1e-5;

#[test]
fn test_avg_pool2d_basic() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 4, 4, 2]),
        false,
        AvgPoolOptions::new([3, 3], [1, 1], [1, 1], false, false),
    );
    run_pool_test(client, 5678, -10.0, 10.0, problem, AVG_POOL2D_TOLERANCE);
}

#[test]
fn test_avg_pool2d_include_pad() {
    let client = TestRuntime::client(&Default::default());
    // count_include_pad = true: divisor is always kernel_size * kernel_size
    let problem = make_problem(
        Shape::from([1, 5, 5, 3]),
        false,
        AvgPoolOptions::new([3, 3], [2, 2], [1, 1], false, true),
    );
    run_pool_test(client, 9999, 0.0, 1.0, problem, AVG_POOL2D_TOLERANCE);
}

#[test]
fn test_avg_pool2d_exclude_pad() {
    let client = TestRuntime::client(&Default::default());
    // count_include_pad = false: divisor only counts non-padded elements
    let problem = make_problem(
        Shape::from([1, 5, 5, 3]),
        false,
        AvgPoolOptions::new([3, 3], [2, 2], [1, 1], false, false),
    );
    run_pool_test(client, 8888, 0.0, 1.0, problem, AVG_POOL2D_TOLERANCE);
}

#[test]
fn test_avg_pool2d_strided_no_pad() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 6, 6, 4]),
        false,
        AvgPoolOptions::new([2, 2], [2, 2], [0, 0], false, false),
    );
    run_pool_test(client, 1234, -1.0, 1.0, problem, AVG_POOL2D_TOLERANCE);
}

#[test]
fn test_avg_pool2d_non_square_asymmetric() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 5, 7, 3]),
        false,
        AvgPoolOptions::new([2, 3], [1, 2], [1, 0], false, false),
    );
    run_pool_test(client, 3456, -1.0, 1.0, problem, AVG_POOL2D_TOLERANCE);
}

#[test]
fn test_avg_pool2d_ceil_mode() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 5, 5, 4]),
        false,
        AvgPoolOptions::new([2, 2], [2, 2], [0, 0], true, false),
    );
    run_pool_test(client, 4567, -1.0, 1.0, problem, AVG_POOL2D_TOLERANCE);
}

#[test]
fn test_avg_pool2d_large_kernel() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([1, 10, 10, 1]),
        false,
        AvgPoolOptions::new([7, 7], [1, 1], [0, 0], false, false),
    );
    run_pool_test(client, 7777, -5.0, 5.0, problem, AVG_POOL2D_TOLERANCE);
}
