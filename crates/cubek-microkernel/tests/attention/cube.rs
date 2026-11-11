//! The cube attention microkernel alone: a cube's planes over one box of queries, of shuffles
//! reading each step's keys where they lie, or of fragments contracting the stages the cube fills.

use crate::launch::attention::{AttentionElements, CubeAttentionLaunch};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::cube::{
    CubeAttentionMicrokernel, CubeAttentionMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Attention, AttentionSettings, DIMS};

impl AttentionSettings for CubeAttentionMicrokernelSettings {
    type Microkernel = CubeAttentionMicrokernel;
    type Launch = CubeAttentionLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: AttentionElements,
        steps: usize,
    ) -> Result<CubeAttentionLaunch, Refusal> {
        CubeAttentionLaunch::new(hw, elements, DIMS, self, steps)
    }
}

/// Every cube attention microkernel the test device runs, at each of its settings, over one and
/// three steps of keys, for `f16` and `f32` inputs summed in `f32` and written as read.
#[test]
fn cube_attention_matches_the_cpu() {
    let attention = Attention::<CubeAttentionMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    attention.check(attention.every_setting());
}
