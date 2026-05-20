use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions};

use super::{make_problem, run_interpolate_test};

const LANCZOS3_TOLERANCE: f32 = 0.00001;

#[test]
fn test_interpolate_lanczos3_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_test(client, 5678, -1.0, 1.0, problem, LANCZOS3_TOLERANCE);
}

#[test]
fn test_interpolate_lanczos3_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_test(client, 1234, -10.0, 10.0, problem, LANCZOS3_TOLERANCE);
}

#[test]
fn test_interpolate_lanczos3_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_test(client, 91011, -100.0, 100.0, problem, LANCZOS3_TOLERANCE);
}

#[test]
fn test_interpolate_lanczos3_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Lanczos3),
    );
    run_interpolate_test(client, 25, -1.0, 1.0, problem, LANCZOS3_TOLERANCE);
}

#[test]
fn test_interpolate_lanczos3_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Lanczos3).with_align_corners(false),
    );
    run_interpolate_test(client, 122, -10.0, 10.0, problem, LANCZOS3_TOLERANCE);
}
