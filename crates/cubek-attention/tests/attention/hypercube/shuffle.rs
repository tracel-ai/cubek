//! The hypercubes of planes of shuffles: each box's keys cut across its cubes, the partials merged
//! by a second launch or by the box's last arrival.

use cubek_attention::kernels::{
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeSecondLaunchShuffleAttentionRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{DECODES, Forward};

/// The second launch's lead on every decode case.
#[test]
fn the_second_launch_routine_matches_the_host() {
    let forward = Forward::new(HypercubeSecondLaunchShuffleAttentionRoutine::new());
    forward.check(forward.at_its_lead(&DECODES));
}

/// The last arrival's lead on every decode case.
#[test]
fn the_last_arrival_routine_matches_the_host() {
    let forward = Forward::new(HypercubeLastArrivalShuffleAttentionRoutine::new());
    forward.check(forward.at_its_lead(&DECODES));
}
