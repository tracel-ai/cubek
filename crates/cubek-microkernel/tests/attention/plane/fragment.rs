//! The plane fragment attention microkernel alone: one plane's queries over every key.

use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionMicrokernelSettings,
};
use cubek_microkernel::testing::attention::{AttentionElements, PlaneFragmentAttentionLaunch};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Attention, AttentionSettings, DIMS};

impl AttentionSettings for PlaneFragmentAttentionMicrokernelSettings {
    type Microkernel = PlaneFragmentAttentionMicrokernel;
    type Launch = PlaneFragmentAttentionLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: AttentionElements,
        steps: usize,
    ) -> Result<PlaneFragmentAttentionLaunch, Refusal> {
        PlaneFragmentAttentionLaunch::new(hw, elements, DIMS, self, steps)
    }
}

/// Every plane fragment attention microkernel the test device runs, at each of its settings, over
/// one and three stages of keys, for `f16` and `f32` inputs summed in `f32`.
#[test]
fn plane_fragment_attention_matches_the_cpu() {
    let attention = Attention::<PlaneFragmentAttentionMicrokernelSettings>::new(
        Hardware::from_client(&cubecl::test_device().client()),
    );
    attention.check(attention.every_setting());
}
