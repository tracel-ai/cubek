//! The plane fragment attention microkernel alone: one plane's queries over every key.

use crate::launch::Lines;
use crate::launch::attention::{AttentionElements, PlaneFragmentAttentionLaunch};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::plane::{
    PlaneFragmentAttentionMicrokernel, PlaneFragmentAttentionMicrokernelSettings,
};
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
        lines: Lines,
        steps: usize,
    ) -> Result<PlaneFragmentAttentionLaunch, Refusal> {
        PlaneFragmentAttentionLaunch::new(hw, elements, DIMS, self, lines, steps)
    }
}

/// Every plane fragment attention microkernel the test device runs, at each of its settings, over
/// one and three stages of keys, for `f16` and `f32` inputs summed in `f32` and written as read.
#[test]
fn plane_fragment_attention_matches_the_cpu() {
    let attention = Attention::<PlaneFragmentAttentionMicrokernelSettings>::new(
        Hardware::from_client(&cubecl::test_device().client()),
    );
    attention.check(attention.every_setting());
}
