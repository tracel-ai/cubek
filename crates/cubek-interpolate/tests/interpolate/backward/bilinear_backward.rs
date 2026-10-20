use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions};

use super::{make_interpolate_backward_problem, run_interpolate_backward_test};

const BILINEAR_BACKWARD_TOLERANCE: f32 = 1e-4;

/// The gather kernel against the forward's scatter adjoint, over one geometry.
///
/// Each case runs both corner conventions, since they invert to different
/// support intervals.
fn check(input_size: [usize; 2], out_grad_shape: [usize; 4], seed: u64) {
    for align_corners in [true, false] {
        let problem = make_interpolate_backward_problem(
            input_size,
            out_grad_shape,
            InterpolateOptions::new(InterpolateMode::Bilinear).with_align_corners(align_corners),
        );
        run_interpolate_backward_test(
            cubecl::test_device().client(),
            seed,
            -1.0,
            1.0,
            problem,
            BILINEAR_BACKWARD_TOLERANCE,
        );
    }
}

#[test]
fn test_interpolate_bilinear_backward_identity() {
    check([4, 4], [2, 4, 4, 16], 5678);
}

#[test]
fn test_interpolate_bilinear_backward_upsample() {
    check([4, 4], [2, 10, 10, 2], 1234);
}

/// A ratio that is not an integer puts output samples at every phase.
#[test]
fn test_interpolate_bilinear_backward_fractional_ratio() {
    check([5, 7], [1, 13, 11, 3], 4321);
}

#[test]
fn test_interpolate_bilinear_backward_downsample() {
    check([9, 9], [2, 4, 3, 4], 91011);
}

/// A singleton input axis takes every output gradient along it.
#[test]
fn test_interpolate_bilinear_backward_singleton() {
    check([1, 6], [1, 5, 4, 8], 2468);
    check([6, 4], [1, 1, 3, 8], 1357);
}
