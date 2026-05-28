use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions, NearestMode};

use super::{make_interpolate_backward_problem, run_interpolate_backward_test};

const NEAREST_BACKWARD_TOLERANCE: f32 = 0.0;

#[test]
fn test_interpolate_nearest_backward_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 4, 4, 16],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(client, 5678, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_exact_backward_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 4, 4, 16],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(client, 5678, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_backward_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 10, 10, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_exact_backward_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 10, 10, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(
        client,
        1234,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_backward_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 2, 2, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_exact_backward_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 2, 2, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(
        client,
        91011,
        -100.0,
        100.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_backward_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 8, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor)),
    );
    run_interpolate_backward_test(client, 25, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_exact_backward_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 8, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact)),
    );
    run_interpolate_backward_test(client, 25, -1.0, 1.0, problem, NEAREST_BACKWARD_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_backward_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 16, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Floor))
            .with_align_corners(false),
    );
    run_interpolate_backward_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}

#[test]
fn test_interpolate_nearest_exact_backward_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_interpolate_backward_problem(
        [4, 4],
        [2, 16, 16, 2],
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact))
            .with_align_corners(false),
    );
    run_interpolate_backward_test(
        client,
        122,
        -10.0,
        10.0,
        problem,
        NEAREST_BACKWARD_TOLERANCE,
    );
}
