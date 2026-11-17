//! The hypercube key gradient microkernel alone: one cube's keys, their gradient and their values' over every query,
//! and its share of the queries' gradient added into a sum.

use crate::launch::attention_backward::HypercubeKeyGradientLaunch;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::AttentionBackwardWorkItem;
use cubek_microkernel::attention_backward::key_gradient::hypercube::{
    HypercubeKeyGradientMicrokernel, HypercubeKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention_backward::{Backward, BackwardSettings};

impl BackwardSettings for HypercubeKeyGradientMicrokernelSettings {
    type Microkernel = HypercubeKeyGradientMicrokernel;
    type Launch = HypercubeKeyGradientLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        steps: usize,
    ) -> Result<HypercubeKeyGradientLaunch, Refusal> {
        HypercubeKeyGradientLaunch::new(hw, item, self, steps)
    }
}

/// Every such microkernel the test device runs, at each of its settings, over one and three steps
/// of its walk, for a forward in `f16` and in `f32`, summed in `f32`, on a device that adds `f32`
/// atomically.
#[test]
fn hypercube_key_gradient_matches_the_host() {
    let backward = Backward::<HypercubeKeyGradientMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    backward.check(backward.every_setting());
}
