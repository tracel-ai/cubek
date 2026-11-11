use cubecl::prelude::Scalar;
use cubek_fft::{FftMode, cfft};
use cubek_test_utils::{
    HostData, HostDataType, HostDataVec, ScopeUnderTest, TestInput, ValidationResult,
    assert_equals_approx,
};

use crate::fft::{CfftRoundTrip, CfftThereAndBack};

/// Scale every element of an f32 `HostData` by `factor`.
fn scaled(mut host: HostData, factor: f32) -> HostData {
    match &mut host.data {
        HostDataVec::F32(values) => values.iter_mut().for_each(|v| *v *= factor),
        _ => panic!("expected f32 host data"),
    }
    host
}

fn combine_re_im(re: ValidationResult, im: ValidationResult) -> ValidationResult {
    use ValidationResult::*;
    match (re, im) {
        (Fail(e), _) | (_, Fail(e)) => Fail(e),
        (Error(e), _) | (_, Error(e)) => Error(e),
        (Skipped(r1), Skipped(r2)) => Skipped(format!("{r1}, {r2}")),
        (Skipped(r), Pass) | (Pass, Skipped(r)) => Skipped(r),
        (Pass, Pass) => Pass,
    }
}

/// Forward then inverse along an axis recovers the input scaled by its length: the transform is
/// unnormalized both ways.
#[test]
fn cfft_roundtrip_axis_last() {
    CfftThereAndBack.check([CfftRoundTrip::of(&[1, 8], 1)]);
}

#[test]
fn cfft_roundtrip_axis_1_strided() {
    CfftThereAndBack.check([CfftRoundTrip::of(&[2, 8, 3], 1)]);
}

#[test]
fn cfft_roundtrip_axis_0_strided() {
    CfftThereAndBack.check([CfftRoundTrip::of(&[16, 2], 0)]);
}

// Batched cases: several windows transformed in one launch, every element
// checked. Guards against per-window state (shared memory) bleeding between
// concurrent cubes, which corrupts all but one window.
#[test]
fn cfft_roundtrip_batched() {
    CfftThereAndBack.check([CfftRoundTrip::of(&[5, 16], 1)]);
}

#[test]
fn cfft_roundtrip_batched_larger() {
    CfftThereAndBack.check([CfftRoundTrip::of(&[4, 64], 1)]);
}

#[test]
fn cfft_wrapper_roundtrip() {
    let client = cubecl::test_device().client();
    let dtype = f32::elem_type_native();
    let signal_shape = [2, 8].to_vec();
    let dim = 1;
    let n = signal_shape[dim] as f32;

    let (input_re, input_re_data) = TestInput::builder(client.clone(), signal_shape.clone())
        .dtype(dtype)
        .uniform(42, -1., 1.)
        .generate_with_f32_host_data();
    let (input_im, input_im_data) = TestInput::builder(client.clone(), signal_shape.clone())
        .dtype(dtype)
        .uniform(7, -1., 1.)
        .generate_with_f32_host_data();

    let (spectrum_re, spectrum_im) =
        cfft(&client, input_re, input_im, dim, dtype, FftMode::Forward);
    let (recovered_re, recovered_im) = cfft(
        &client,
        spectrum_re,
        spectrum_im,
        dim,
        dtype,
        FftMode::Inverse,
    );

    let actual_re = HostData::from_tensor_handle(&client, recovered_re, HostDataType::F32);
    let actual_im = HostData::from_tensor_handle(&client, recovered_im, HostDataType::F32);
    combine_re_im(
        assert_equals_approx(&actual_re, &scaled(input_re_data, n), 1e-2),
        assert_equals_approx(&actual_im, &scaled(input_im_data, n), 1e-2),
    )
    .as_test_outcome()
    .enforce();
}
