//! The bilinear backward: the gather kernel against the forward's scatter adjoint, each geometry
//! under both corner conventions, since they invert to different support intervals.

use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions};
use cubek_test_utils::ScopeUnderTest;

use crate::interpolate::backward::{Backward, BackwardCase};

/// Bilinear, corners aligned: the options every geometry starts from.
const BILINEAR: InterpolateOptions = InterpolateOptions {
    mode: InterpolateMode::Bilinear,
    align_corners: true,
};

/// Every geometry.
const GEOMETRIES: [BackwardCase; 6] = [
    // An identity, an upsample and a downsample.
    BackwardCase {
        options: BILINEAR,
        input: [4, 4],
        out_gradient: [2, 4, 4, 16],
        seed: 5678,
        range: [-1.0, 1.0],
    },
    BackwardCase {
        options: BILINEAR,
        input: [4, 4],
        out_gradient: [2, 10, 10, 2],
        seed: 1234,
        range: [-1.0, 1.0],
    },
    BackwardCase {
        options: BILINEAR,
        input: [9, 9],
        out_gradient: [2, 4, 3, 4],
        seed: 91011,
        range: [-1.0, 1.0],
    },
    // A ratio that is not an integer puts output samples at every phase.
    BackwardCase {
        options: BILINEAR,
        input: [5, 7],
        out_gradient: [1, 13, 11, 3],
        seed: 4321,
        range: [-1.0, 1.0],
    },
    // A singleton input axis takes every output gradient along it.
    BackwardCase {
        options: BILINEAR,
        input: [1, 6],
        out_gradient: [1, 5, 4, 8],
        seed: 2468,
        range: [-1.0, 1.0],
    },
    BackwardCase {
        options: BILINEAR,
        input: [6, 4],
        out_gradient: [1, 1, 3, 8],
        seed: 1357,
        range: [-1.0, 1.0],
    },
];

/// Every geometry under both corner conventions.
#[test]
fn every_bilinear_geometry_matches_the_reference() {
    let cases: Vec<_> = GEOMETRIES
        .into_iter()
        .flat_map(|case| {
            case.under([true, false].map(|align_corners| InterpolateOptions {
                align_corners,
                ..BILINEAR
            }))
        })
        .collect();
    Backward.check(cases);
}
