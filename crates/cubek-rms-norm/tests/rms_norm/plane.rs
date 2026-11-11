//! The plane routine: a row a plane, held in its lanes' registers.

use cubek_rms_norm::kernels::PlaneRmsNormRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::rms_norm::{Declines, RmsNorm};

/// A plane serves every row its lanes hold, and declines the rest by its microkernel's reasons.
#[test]
fn plane_routine_matches_the_reference() {
    let rms_norm = RmsNorm::new(PlaneRmsNormRoutine, Declines::AsItsMicrokernelStates);
    rms_norm.check(rms_norm.every_case());
}
