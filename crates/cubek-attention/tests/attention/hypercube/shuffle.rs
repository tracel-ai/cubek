//! The hypercubes of planes of shuffles: each box's keys cut across its cubes, the partials merged
//! by a second launch or by the box's last arrival.

use cubek_attention::kernels::{
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeSecondLaunchShuffleAttentionRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{DECODES, DECODES_WIDE_FP8, Forward};

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

/// Every setting of both merges over a decode step of every FP8 cache: the planes widen the
/// `e4m3` keys or values into the sum as they read them.
#[test]
fn every_hypercube_shuffle_setting_reads_every_fp8_cache() {
    let second_launch = Forward::new(HypercubeSecondLaunchShuffleAttentionRoutine::new());
    second_launch.check(second_launch.at_every_setting(&DECODES_WIDE_FP8));
    let last_arrival = Forward::new(HypercubeLastArrivalShuffleAttentionRoutine::new());
    last_arrival.check(last_arrival.at_every_setting(&DECODES_WIDE_FP8));
}
