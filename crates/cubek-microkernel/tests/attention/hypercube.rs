//! The hypercube attention microkernel alone: cubes sharing one box of queries' keys, each walking
//! its run into its slot, the last of them to arrive merging the slots.

use crate::launch::attention::{AttentionElements, HypercubeAttentionLaunch};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::cube::CubeAttentionPlaneSettings;
use cubek_microkernel::attention::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Attention, AttentionSettings, DIMS};

impl AttentionSettings for HypercubeAttentionMicrokernelSettings {
    type Microkernel = HypercubeAttentionMicrokernel;
    type Launch = HypercubeAttentionLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: AttentionElements,
        steps: usize,
    ) -> Result<HypercubeAttentionLaunch, Refusal> {
        HypercubeAttentionLaunch::new(hw, elements, DIMS, self, steps)
    }
}

/// Every hypercube attention microkernel of plane shuffles the test device runs, at each of its
/// settings, over one step of keys, which one of three cubes walks, and three, which the three
/// share; for `f16` and `f32` inputs summed in `f32` and written as read. Planes of fragments leave
/// slots a second launch merges, not the last arrival this tests.
#[test]
fn hypercube_attention_matches_the_cpu() {
    let attention = Attention::<HypercubeAttentionMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    let shuffles = attention.every_setting().into_iter().filter(|case| {
        matches!(
            case.settings.cube.plane,
            CubeAttentionPlaneSettings::Shuffle(_)
        )
    });
    attention.check(shuffles);
}
