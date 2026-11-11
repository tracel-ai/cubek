//! Every RMS-norm routine, at every setting it lists, against an `f64` host reference.

use super::check::{Declines, check_routine};
use cubek_rms_norm::kernels::{CubeGroupsRmsNormRoutine, CubeRmsNormRoutine, PlaneRmsNormRoutine};

/// The floor serves every case at every setting: a decline would be a family with nothing to fall
/// back on.
#[test]
fn cube_groups_routine_matches_the_reference() {
    check_routine(CubeGroupsRmsNormRoutine, Declines::Never);
}

/// A plane serves every row its lanes hold, and declines the rest by its microkernel's reasons.
#[test]
fn plane_routine_matches_the_reference() {
    check_routine(PlaneRmsNormRoutine, Declines::AsItsMicrokernelStates);
}

/// A cube serves every row its planes' runs fit, and declines the rest by its microkernel's
/// reasons.
#[test]
fn cube_routine_matches_the_reference() {
    check_routine(CubeRmsNormRoutine, Declines::AsItsMicrokernelStates);
}
