use super::{build_output_shape, build_output_tensor, output_host_f32, validate_test};
use crate::nearest::reference_nearest;
use cubecl::{TestRuntime, prelude::*};
use cubek_interpolate::interpolate;
use cubek_interpolate::interpolate_options::{InterpolateMode, InterpolateOptions};
use cubek_test_utils::TestInput;

const NEAREST_TOLERANCE: f32 = 0.;

#[test]
fn test_interpolate_nearest_identity() {
    let client = TestRuntime::client(&Default::default());
    let (input, input_data) = TestInput::builder(client.clone(), vec![2, 4, 4, 2])
        .uniform(5678, -10.0, 10.0)
        .generate_with_f32_host_data();

    let output_size: [usize; 2] = [4, 4];
    let output_shape = build_output_shape(&input_data, output_size);
    let output = build_output_tensor(&client, output_shape.clone(), input.dtype);
    let result = interpolate(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        InterpolateOptions::new(InterpolateMode::Nearest),
        input.dtype,
    );

    let output_host = output_host_f32(&client, output);
    let reference = reference_nearest(&input_data, &output_shape);

    validate_test(
        result.clone(),
        output_host.clone(),
        reference,
        NEAREST_TOLERANCE,
    );

    // Check that the output matches the input since it's an identity resize
    validate_test(result, output_host, input_data, NEAREST_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_upsample() {
    let client = TestRuntime::client(&Default::default());
    let (input, input_data) = TestInput::builder(client.clone(), vec![2, 4, 4, 2])
        .uniform(1234, -10.0, 10.0)
        .generate_with_f32_host_data();

    let output_size: [usize; 2] = [10, 10];
    let output_shape = build_output_shape(&input_data, output_size);
    let output = build_output_tensor(&client, output_shape.clone(), input.dtype);
    let result = interpolate(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        InterpolateOptions::new(InterpolateMode::Nearest),
        input.dtype,
    );

    let output_host = output_host_f32(&client, output);
    let reference = reference_nearest(&input_data, &output_shape);

    validate_test(result, output_host, reference, NEAREST_TOLERANCE);
}

#[test]
fn test_interpolate_nearest_downsample() {
    let client = TestRuntime::client(&Default::default());
    let (input, input_data) = TestInput::builder(client.clone(), vec![2, 10, 10, 2])
        .uniform(91011, -10.0, 10.0)
        .generate_with_f32_host_data();

    let output_size: [usize; 2] = [2, 2];
    let output_shape = build_output_shape(&input_data, output_size);
    let output = build_output_tensor(&client, output_shape.clone(), input.dtype);
    let result = interpolate(
        &client,
        input.clone().binding(),
        output.clone().binding(),
        InterpolateOptions::new(InterpolateMode::Nearest),
        input.dtype,
    );

    let output_host = output_host_f32(&client, output);
    let reference = reference_nearest(&input_data, &output_shape);

    validate_test(result, output_host, reference, NEAREST_TOLERANCE);
}
