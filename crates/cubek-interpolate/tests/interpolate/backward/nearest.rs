//! The nearest backward: every output gradient summed into the input cell its sample rounds to,
//! under both roundings.

use cubek_interpolate::definition::{InterpolateMode, InterpolateOptions, NearestMode};
use cubek_test_utils::ScopeUnderTest;

use crate::interpolate::backward::{Backward, BackwardCase};

/// Flooring, corners aligned: the options every geometry starts from.
const FLOOR: InterpolateOptions = InterpolateOptions {
    mode: InterpolateMode::Nearest(NearestMode::Floor),
    align_corners: true,
};

/// Every geometry, flooring.
const GEOMETRIES: [BackwardCase; 5] = [
    // An identity, an upsample, a downsample and a resize to another aspect.
    BackwardCase {
        options: FLOOR,
        input: [4, 4],
        out_gradient: [2, 4, 4, 16],
        seed: 5678,
        range: [-1.0, 1.0],
    },
    BackwardCase {
        options: FLOOR,
        input: [4, 4],
        out_gradient: [2, 10, 10, 2],
        seed: 1234,
        range: [-10.0, 10.0],
    },
    BackwardCase {
        options: FLOOR,
        input: [4, 4],
        out_gradient: [2, 2, 2, 2],
        seed: 91011,
        range: [-100.0, 100.0],
    },
    BackwardCase {
        options: FLOOR,
        input: [4, 4],
        out_gradient: [2, 8, 16, 2],
        seed: 25,
        range: [-1.0, 1.0],
    },
    // The other coordinate transform.
    BackwardCase {
        options: InterpolateOptions {
            align_corners: false,
            ..FLOOR
        },
        input: [4, 4],
        out_gradient: [2, 16, 16, 2],
        seed: 122,
        range: [-10.0, 10.0],
    },
];

/// Every geometry under both roundings, flooring and exact.
#[test]
fn every_nearest_geometry_matches_the_reference() {
    let cases: Vec<_> = GEOMETRIES
        .into_iter()
        .flat_map(|case| {
            case.under([NearestMode::Floor, NearestMode::Exact].map(|rounding| {
                InterpolateOptions {
                    mode: InterpolateMode::Nearest(rounding),
                    ..case.options
                }
            }))
        })
        .collect();
    Backward.check(cases);
}
