//! The cube staged microkernel alone: a cube's units fill its stages, its planes contract them.

use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::cube::{CubeStagedMicrokernel, CubeStagedMicrokernelSettings};
use cubek_microkernel::testing::matmul::{CubeStagedLaunch, MatmulElements};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Matmul, MatmulSettings};

impl MatmulSettings for CubeStagedMicrokernelSettings {
    type Microkernel = CubeStagedMicrokernel;
    type Launch = CubeStagedLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        steps: usize,
    ) -> Result<CubeStagedLaunch, Refusal> {
        CubeStagedLaunch::new(hw, elements, self, steps)
    }
}

/// Every cube staged microkernel the test device runs, at each of its settings, over one and three
/// of its stages of `K`, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn cube_staged_matmul_matches_the_cpu() {
    let matmul = Matmul::<CubeStagedMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    matmul.check(matmul.every_setting());
}
