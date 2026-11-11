//! The plane fragment microkernel alone: one plane walks its tile through its own stages.

use crate::launch::matmul::{MatmulElements, PlaneFragmentLaunch};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::matmul::plane::{
    PlaneFragmentMicrokernel, PlaneFragmentMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::matmul::{Matmul, MatmulSettings};

impl MatmulSettings for PlaneFragmentMicrokernelSettings {
    type Microkernel = PlaneFragmentMicrokernel;
    type Launch = PlaneFragmentLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: MatmulElements,
        steps: usize,
    ) -> Result<PlaneFragmentLaunch, Refusal> {
        PlaneFragmentLaunch::new(hw, elements, self, steps)
    }
}

/// Every plane fragment microkernel the test device runs, at each of its settings, over one and
/// three stages of `K`, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn plane_fragment_matmul_matches_the_cpu() {
    let matmul = Matmul::<PlaneFragmentMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    matmul.check(matmul.every_setting());
}
