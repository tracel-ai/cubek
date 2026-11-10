//! The hypercube of plane fragments: each box's keys cut across its cubes, a second launch merging
//! the partials they leave.

use cubek_attention::kernels::HypercubeSecondLaunchFragmentAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Forward, PROMPT_WIDE, PROMPTS};

/// The routine's lead on every prompt case.
#[test]
fn the_hypercube_fragment_routine_matches_the_host() {
    let forward = Forward::new(HypercubeSecondLaunchFragmentAttentionRoutine);
    forward.check(forward.at_its_lead(&PROMPTS));
}

/// Every setting the routine lists at a model's head dim: the stages each cube hands on to the
/// partial it leaves are each a different kernel.
#[test]
fn every_hypercube_fragment_setting_matches_the_host_at_a_head_dim_of_128() {
    let forward = Forward::new(HypercubeSecondLaunchFragmentAttentionRoutine);
    forward.check(forward.at_every_setting(&[PROMPT_WIDE]));
}
