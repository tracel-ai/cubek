use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions};

use super::{make_problem, run_interpolate_test};

const BICUBIC_TOLERANCE: f32 = 0.00001;

#[test]
fn test_interpolate_bicubic_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 16],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_test(client, 5678, -1.0, 1.0, problem, BICUBIC_TOLERANCE);
}

#[test]
fn test_interpolate_bicubic_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_test(client, 1234, -10.0, 10.0, problem, BICUBIC_TOLERANCE);
}

#[test]
fn test_interpolate_bicubic_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_test(client, 91011, -100.0, 100.0, problem, BICUBIC_TOLERANCE);
}

#[test]
fn test_interpolate_bicubic_resize() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [8, 16],
        InterpolateOptions::new(InterpolateMode::Bicubic),
    );
    run_interpolate_test(client, 25, -1.0, 1.0, problem, BICUBIC_TOLERANCE);
}

#[test]
fn test_interpolate_bicubic_without_align_corners() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [16, 16],
        InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false),
    );
    run_interpolate_test(client, 122, -10.0, 10.0, problem, BICUBIC_TOLERANCE);
}
