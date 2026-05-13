use cubecl::prelude::*;
use cubek_interpolate::{components::mode::Interpolate, components::mode::Nearest};

#[test]
fn test_interpolate_nearest_identity() {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem(
        [2, 4, 4, 2],
        [4, 4],
        InterpolateOptions::new(InterpolateMode::Nearest),
    );
    run_interpolate_test(client, 5678, -1.0, 1.0, problem, NEAREST_TOLERANCE);
}
