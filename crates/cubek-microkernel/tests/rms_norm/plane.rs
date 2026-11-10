//! The plane RMS norm microkernel alone: one plane holds its row in its lanes' registers.

use cubek_microkernel::rms_norm::plane::PlaneRmsNormMicrokernel;
use cubek_microkernel::testing::rms_norm::RmsNormSettings;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::ScopeUnderTest;

use crate::rms_norm::{RmsNorm, RmsNormCase};

/// A plane normalizes every row as the host does, at each of its settings.
#[test]
fn a_plane_normalizes_every_row_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let settings = PlaneRmsNormMicrokernel::settings(&hw)
        .into_iter()
        .map(RmsNormSettings::Plane);
    RmsNorm { hw }.check(RmsNormCase::every_shape(settings));
}
