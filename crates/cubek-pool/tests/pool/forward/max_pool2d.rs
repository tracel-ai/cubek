use super::{make_problem, run_pool_test};
use cubecl::{Runtime, TestRuntime, zspace::Shape};
use cubek_pool::definition::MaxPoolOptions;

const MAX_POOL2D_TOLERANCE: f32 = 0.0;

#[test]
fn test_max_pool2d() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 4, 4, 2]),
        false,
        MaxPoolOptions::new([3, 3], [1, 1], [1, 1], [1, 1], false),
    );
    run_pool_test(client, 5678, -10.0, 10.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_max_pool2d_strided_no_pad() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 6, 6, 4]),
        false,
        MaxPoolOptions::new([2, 2], [2, 2], [0, 0], [1, 1], false),
    );
    run_pool_test(client, 1234, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_max_pool2d_dilated() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([1, 8, 8, 3]),
        false,
        MaxPoolOptions::new([3, 3], [1, 1], [0, 0], [2, 2], false),
    );
    run_pool_test(client, 2345, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_max_pool2d_non_square_asymmetric() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 5, 7, 3]),
        false,
        MaxPoolOptions::new([2, 3], [1, 2], [1, 0], [1, 1], false),
    );
    run_pool_test(client, 3456, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_max_pool2d_ceil_mode() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([2, 5, 5, 4]),
        false,
        MaxPoolOptions::new([2, 2], [2, 2], [0, 0], [1, 1], true),
    );
    run_pool_test(client, 4567, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_max_pool2d_with_indices() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        Shape::from([1, 3, 3, 2]),
        true,
        MaxPoolOptions::new([2, 2], [1, 1], [0, 0], [1, 1], false),
    );
    run_pool_test(client, 6789, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}
