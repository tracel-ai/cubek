//! Validation of the experimental tile-backed forward path.

use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::{
    definition::{InterpolateMode, InterpolateOptions, NearestMode},
    eval::cpu_reference::cpu_reference_interpolate_from_host,
    interpolate_tile,
};
use cubek_test_utils::{TestInput, assert_equals_approx};

use super::{build_output_tensor, make_problem, output_host_f32, validate_test};

const TOLERANCE: f32 = 0.0001;

fn tile_output(options: InterpolateOptions) {
    let client = TestRuntime::client(&Default::default());
    let problem = make_problem([2, 8, 9, 4], [13, 15], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(123, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected =
        cpu_reference_interpolate_from_host(&input_data, &problem.output_shape(), &options);
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);
    let result = interpolate_tile(
        &client,
        input.binding(),
        output.clone().binding(),
        options,
        output.dtype,
    );
    let actual = output_host_f32(&client, output);
    validate_test(result, actual, expected, TOLERANCE);
}

#[test]
fn test_interpolate_tile_nearest_exact() {
    tile_output(
        InterpolateOptions::new(InterpolateMode::Nearest(NearestMode::Exact))
            .with_align_corners(false),
    );
}

#[test]
fn test_interpolate_tile_bicubic() {
    tile_output(InterpolateOptions::new(InterpolateMode::Bicubic).with_align_corners(false));
}

#[test]
fn test_interpolate_tile_lanczos3_identity() {
    let client = TestRuntime::client(&Default::default());
    let options = InterpolateOptions::new(InterpolateMode::Lanczos3);
    let problem = make_problem([2, 8, 9, 4], [8, 9], options);
    let (input, input_data) = TestInput::builder(client.clone(), problem.input_shape())
        .uniform(123, -3.0, 3.0)
        .generate_with_f32_host_data();
    let expected =
        cpu_reference_interpolate_from_host(&input_data, &problem.output_shape(), &options);
    let output = build_output_tensor(&client, problem.output_shape().to_vec(), input.dtype);
    let result = interpolate_tile(
        &client,
        input.binding(),
        output.clone().binding(),
        options,
        output.dtype,
    );
    let actual = output_host_f32(&client, output);

    result.unwrap();
    // At integral coordinates every non-central Lanczos weight is zero, so this validates the
    // six-tap path without exercising its intentionally unnormalized border behavior.
    assert_equals_approx(&actual, &expected, TOLERANCE)
        .as_test_outcome()
        .enforce();
}
