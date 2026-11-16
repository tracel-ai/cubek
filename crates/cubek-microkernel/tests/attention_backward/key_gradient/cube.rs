//! The cube key gradient microkernel alone: one cube's keys, their gradient and their values' over every query.

use crate::launch::attention_backward::CubeKeyGradientLaunch;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::AttentionBackwardWorkItem;
use cubek_microkernel::attention_backward::key_gradient::cube::{
    CubeKeyGradientMicrokernel, CubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention_backward::{Backward, BackwardSettings};

impl BackwardSettings for CubeKeyGradientMicrokernelSettings {
    type Microkernel = CubeKeyGradientMicrokernel;
    type Launch = CubeKeyGradientLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        steps: usize,
    ) -> Result<CubeKeyGradientLaunch, Refusal> {
        CubeKeyGradientLaunch::new(hw, item, self, steps)
    }
}

/// Every such microkernel the test device runs, at each of its settings, over one and three steps
/// of its walk, for a forward in `f16` and in `f32`, summed in `f32`.
#[test]
fn cube_key_gradient_matches_the_host() {
    let backward = Backward::<CubeKeyGradientMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    backward.check(backward.every_setting());
}
