//! The cube groups RMS norm microkernel alone: a group of a cube's units shares one row.

use crate::launch::rms_norm::RmsNormSettings;
use cubek_microkernel::rms_norm::cube::CubeGroupsRmsNormMicrokernel;
use cubek_std::Hardware;
use cubek_std::microkernel::Microkernel;
use cubek_test_utils::ScopeUnderTest;

use crate::rms_norm::{RmsNorm, RmsNormCase};

/// A group normalizes every row as the host does, at each of its settings.
#[test]
fn a_group_normalizes_every_row_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let settings = CubeGroupsRmsNormMicrokernel::settings(&hw)
        .into_iter()
        .map(RmsNormSettings::CubeGroups);
    RmsNorm { hw }.check(RmsNormCase::every_shape(settings));
}

/// A group gates its rows before the norm and after it, and writes them wider than it reads them,
/// as the host does, at each of its settings.
#[test]
fn a_group_gates_and_widens_every_row_like_the_host() {
    let hw = Hardware::from_client(&cubecl::test_device().client());
    let settings = CubeGroupsRmsNormMicrokernel::settings(&hw)
        .into_iter()
        .map(RmsNormSettings::CubeGroups);
    RmsNorm { hw }.check(RmsNormCase::gated_and_widened(settings));
}
