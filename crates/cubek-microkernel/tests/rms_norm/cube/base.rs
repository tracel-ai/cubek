//! The cube RMS norm microkernel alone: a cube's planes each hold a run of one row, their sums met
//! through shared memory.

use crate::launch::rms_norm::RmsNormSettings;
use cubek_microkernel::rms_norm::cube::CubeRmsNormMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::ScopeUnderTest;

use crate::rms_norm::{RmsNorm, RmsNormCase};

/// A cube normalizes every row as the host does, at each of its settings.
#[test]
fn a_cube_normalizes_every_row_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let settings = CubeRmsNormMicrokernel::settings(&hw)
        .into_iter()
        .map(RmsNormSettings::Cube);
    RmsNorm { hw }.check(RmsNormCase::every_shape(settings));
}
