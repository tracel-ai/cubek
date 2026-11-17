//! The plane query gradient microkernel alone: one plane's queries, their dot products and their gradient over every key.

use crate::launch::attention_backward::PlaneQueryGradientLaunch;
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention_backward::AttentionBackwardWorkItem;
use cubek_microkernel::attention_backward::query_gradient::plane::{
    PlaneQueryGradientMicrokernel, PlaneQueryGradientMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention_backward::{Backward, BackwardSettings};

impl BackwardSettings for PlaneQueryGradientMicrokernelSettings {
    type Microkernel = PlaneQueryGradientMicrokernel;
    type Launch = PlaneQueryGradientLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        item: AttentionBackwardWorkItem,
        steps: usize,
    ) -> Result<PlaneQueryGradientLaunch, Refusal> {
        PlaneQueryGradientLaunch::new(hw, item, self, steps)
    }
}

/// Every such microkernel the test device runs, at each of its settings, over one and three steps
/// of its walk, for a forward in `f16` and in `f32`, summed in `f32`.
#[test]
fn plane_query_gradient_matches_the_host() {
    let backward = Backward::<PlaneQueryGradientMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    backward.check(backward.every_setting());
}
