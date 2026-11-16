//! The cube routine: a row a cube, its planes each a run of it.

use cubek_rms_norm::kernels::CubeRmsNormRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::rms_norm::{Declines, RmsNorm};

/// A cube serves every row its planes' runs fit, and declines the rest by its microkernel's
/// reasons.
#[test]
fn cube_routine_matches_the_reference() {
    let rms_norm = RmsNorm::new(CubeRmsNormRoutine, Declines::AsItsMicrokernelStates);
    rms_norm.check(rms_norm.every_case());
}
