//! The unit key gradient microkernel alone: one unit's keys, their gradient and their values' over every query.

use crate::launch::attention_backward::UnitKeyGradientLaunch;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::AttentionBackwardWorkItem;
use cubek_microkernel::attention_backward::key_gradient::unit::{
    UnitKeyGradientMicrokernel, UnitKeyGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention_backward::{Backward, BackwardSettings};

impl BackwardSettings for UnitKeyGradientMicrokernelSettings {
    type Microkernel = UnitKeyGradientMicrokernel;
    type Launch = UnitKeyGradientLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        steps: usize,
    ) -> Result<UnitKeyGradientLaunch, Refusal> {
        UnitKeyGradientLaunch::new(hw, item, self, steps)
    }
}

/// Every such microkernel the test device runs, at each of its settings, over one and three steps
/// of its walk, for a forward in `f16` and in `f32`, summed in `f32`.
#[test]
fn unit_key_gradient_matches_the_host() {
    let backward = Backward::<UnitKeyGradientMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    backward.check(backward.every_setting());
}
