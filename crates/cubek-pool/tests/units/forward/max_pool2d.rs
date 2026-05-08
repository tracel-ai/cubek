use cubecl::{Runtime, TestRuntime};

use super::{make_max_pool2d_problem, run_max_pool2d_test};

const MAX_POOL2D_TOLERANCE: f32 = 0.0;

#[test]
fn test_pool_max_2d() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_max_pool2d_problem([2, 4, 4, 2], [3, 3], [1, 1], [1, 1], [1, 1], [false; 2]);
    run_max_pool2d_test(client, 5678, -10.0, 10.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_pool_max_2d_strided_no_pad() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_max_pool2d_problem([2, 6, 6, 4], [2, 2], [2, 2], [0, 0], [1, 1], [false; 2]);
    run_max_pool2d_test(client, 1234, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_pool_max_2d_dilated() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_max_pool2d_problem([1, 8, 8, 3], [3, 3], [1, 1], [0, 0], [2, 2], [false; 2]);
    run_max_pool2d_test(client, 2345, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_pool_max_2d_non_square_asymmetric() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_max_pool2d_problem([2, 5, 7, 3], [2, 3], [1, 2], [1, 0], [1, 1], [false; 2]);
    run_max_pool2d_test(client, 3456, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}

#[test]
fn test_pool_max_2d_ceil_mode() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_max_pool2d_problem([2, 5, 5, 4], [2, 2], [2, 2], [0, 0], [1, 1], [true; 2]);
    run_max_pool2d_test(client, 4567, -1.0, 1.0, problem, MAX_POOL2D_TOLERANCE);
}
