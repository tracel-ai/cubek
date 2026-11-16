//! The unit attention microkernel alone: one unit's query rows over every key.

use crate::launch::Lines;
use crate::launch::attention::{AttentionElements, UnitAttentionLaunch};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::unit::{
    UnitAttentionMicrokernel, UnitAttentionMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Attention, AttentionSettings, DIMS};

impl AttentionSettings for UnitAttentionMicrokernelSettings {
    type Microkernel = UnitAttentionMicrokernel;
    type Launch = UnitAttentionLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: AttentionElements,
        lines: Lines,
        steps: usize,
    ) -> Result<UnitAttentionLaunch, Refusal> {
        UnitAttentionLaunch::new(hw, elements, DIMS, self, lines, steps)
    }
}

/// Every unit attention microkernel the test device runs, at each of its settings, over one and
/// three steps of keys, for `f16` and `f32` inputs summed in `f32` and written as read.
#[test]
fn unit_attention_matches_the_cpu() {
    let attention = Attention::<UnitAttentionMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    attention.check(attention.every_setting());
}
