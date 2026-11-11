//! The hypercube attention microkernel alone: cubes sharing one box of queries' keys, each walking
//! its run into its slot, the last of them to arrive merging the slots.

use crate::launch::Lines;
use crate::launch::attention::{AttentionElements, HypercubeAttentionLaunch};
use cubecl::tile::launch::Refusal;
use cubek_microkernel::attention::AttentionVisibility;
use cubek_microkernel::attention::cube::CubeAttentionPlaneSettings;
use cubek_microkernel::attention::hypercube::{
    HypercubeAttentionMicrokernel, HypercubeAttentionMicrokernelSettings,
};
use cubek_std::Hardware;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Attention, AttentionCase, AttentionSettings, DIMS};

impl AttentionSettings for HypercubeAttentionMicrokernelSettings {
    type Microkernel = HypercubeAttentionMicrokernel;
    type Launch = HypercubeAttentionLaunch;

    fn launch(
        &self,
        hw: &Hardware,
        elements: AttentionElements,
        lines: Lines,
        steps: usize,
    ) -> Result<HypercubeAttentionLaunch, Refusal> {
        HypercubeAttentionLaunch::new(hw, elements, DIMS, self, lines, steps)
    }
}

/// Every hypercube attention microkernel of plane shuffles the test device runs, at each of its
/// settings, its keys shared by three cubes: one step long, where two cubes walk nothing; three,
/// one step each; and four and five, where the runs differ in length. For `f16` and `f32` inputs
/// summed in `f32` and written as read, every key attended; and over three and five steps under
/// each visibility that hides keys, where a cube's whole run may lie past what its queries see.
/// Planes of fragments leave slots a second launch merges, not the last arrival this tests.
#[test]
fn hypercube_attention_matches_the_cpu() {
    let attention = Attention::<HypercubeAttentionMicrokernelSettings>::new(Hardware::from_client(
        &cubecl::test_device().client(),
    ));
    let cases: Vec<_> = attention
        .every_setting()
        .into_iter()
        .filter(|case| {
            matches!(
                case.settings.cube.plane,
                CubeAttentionPlaneSettings::Shuffle(_)
            )
        })
        .flat_map(|case| {
            let steps: &[usize] = match (case.visibility, case.steps) {
                (AttentionVisibility::Attended, 1) => &[1, 3, 4, 5],
                (AttentionVisibility::Attended, _) => &[],
                _ => &[3, 5],
            };
            steps
                .iter()
                .map(move |&steps| AttentionCase { steps, ..case })
        })
        .collect();
    attention.check(cases);
}
