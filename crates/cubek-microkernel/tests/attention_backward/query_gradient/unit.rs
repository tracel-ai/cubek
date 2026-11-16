//! The unit query gradient microkernel alone: one unit's queries, their dot products and their gradient over every key.

use crate::launch::attention_backward::UnitQueryGradientLaunch;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::AttentionBackwardWorkItem;
use cubek_microkernel::attention_backward::query_gradient::unit::{
    UnitQueryGradientMicrokernel, UnitQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention_backward::{Backward, BackwardSettings};

impl BackwardSettings for UnitQueryGradientMicrokernelSettings {
    type Microkernel = UnitQueryGradientMicrokernel;
    type Launch = UnitQueryGradientLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        steps: usize,
    ) -> Result<UnitQueryGradientLaunch, Refusal> {
        UnitQueryGradientLaunch::new(hw, item, self, steps)
    }
}

/// Every such microkernel the test device runs, at each of its settings, over one and three steps
/// of its walk, for a forward in `f16` and in `f32`, summed in `f32`.
#[test]
fn unit_query_gradient_matches_the_host() {
    let backward = Backward::<UnitQueryGradientMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    backward.check(backward.every_setting());
}
