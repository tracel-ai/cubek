//! The hypercubes of planes of shuffles: each box's keys cut across its cubes, the partials merged
//! by the box's last arrival, or by the split-keys merge launched after the partial routine.

use cubek_attention::kernels::{
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeShufflePartialAttentionRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{DECODES, Forward};

/// The partial routine's lead, merged by a second launch, on every decode case.
#[test]
fn the_partial_routine_merged_matches_the_host() {
    let forward = Forward::split_keys(HypercubeShufflePartialAttentionRoutine::new());
    forward.check(forward.at_its_lead(&DECODES));
}

/// The last arrival's lead on every decode case.
#[test]
fn the_last_arrival_routine_matches_the_host() {
    let forward = Forward::new(HypercubeLastArrivalShuffleAttentionRoutine::new());
    forward.check(forward.at_its_lead(&DECODES));
}
