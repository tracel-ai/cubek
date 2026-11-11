//! The hypercube staged microkernel alone: cubes sharing one box's `K`, each walking its run of
//! the stages, the box's last arrival adding their parts up.

use crate::launch::matmul::{HypercubeStagedLaunch, MatmulElements};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::hypercube::{
    HypercubeStagedMicrokernel, HypercubeStagedMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Matmul, MatmulSettings};

impl MatmulSettings for HypercubeStagedMicrokernelSettings {
    type Microkernel = HypercubeStagedMicrokernel;
    type Launch = HypercubeStagedLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        steps: usize,
    ) -> Result<HypercubeStagedLaunch, Refusal> {
        HypercubeStagedLaunch::new(hw, elements, self, steps)
    }
}

/// Every hypercube staged microkernel the test device runs, at each of its settings, over one of
/// its stages of `K`, which the first of three cubes holds whole, and three, which the three cut
/// into parts the last to arrive adds up; for `f16` and `f32` inputs summed in `f32`.
#[test]
fn hypercube_staged_matmul_matches_the_cpu() {
    let matmul = Matmul::<HypercubeStagedMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    matmul.check(matmul.every_setting());
}
