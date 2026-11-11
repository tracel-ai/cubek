//! A product grouped along `K`: every group sums its own run of rows, as a weight gradient of a
//! mixture of experts sums each expert's tokens. An empty run answers zeros.

use cubek_matmul::kernels::CubeStagedMatmulRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::Declines;
use crate::tiled::ByRoutine;
use crate::tiled::indexed::RunsOfK;

#[test]
fn every_cube_staged_setting_sums_each_group_s_run() {
    let staged = ByRoutine::new(CubeStagedMatmulRoutine, Declines::AsItStates);
    let runs = RunsOfK {
        m: 64,
        n: 64,
        ends: vec![37, 37, 100],
    };
    staged.check(staged.at_every_setting(&[runs]));
}
