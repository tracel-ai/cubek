use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions};

use super::{make_problem, run_test};

const NEAREST_TOLERANCE: f32 = 0.;

#[test]
fn test_interpolate_nearest_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    run_test(client, 5678, -1.0, 1.0, problem, NEAREST_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_upsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [10, 10],
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    run_test(client, 1234, -10.0, 10.0, problem, NEAREST_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_downsample() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [2, 2],
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    run_test(client, 91011, -100.0, 100.0, problem, NEAREST_TOLERANCE);
}
