//! The cube staged routine: a cube's planes of fragments contracting the stages it fills.

use cubek_attention::kernels::CubeStagedAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Forward, PROMPT, PROMPT_WIDE, PROMPTS};

/// The cube staged routine's lead on every prompt case.
#[test]
fn the_cube_staged_routine_matches_the_host() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_its_lead(&PROMPTS));
}

/// Planes that split the same rows' keys merge their state before they write it.
#[test]
fn planes_splitting_the_keys_match_the_host() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    let split = forward
        .settings()
        .into_iter()
        .filter(|settings| settings.microkernel.planes_along_keys.get() > 1)
        .take(1);
    forward.check(forward.at(split, &[PROMPT]));
}

/// Every setting the routine lists, the copy engine's among them, at a model's head dim: the
/// query's and the stages' layouts, the rings and the run of score fragments the value reads are
/// each a different kernel.
#[test]
fn every_cube_staged_setting_matches_the_host_at_a_head_dim_of_128() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_every_setting(&[PROMPT_WIDE]));
}
