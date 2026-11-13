//! The hypercubes of planes of shuffles: each box's keys cut across its cubes, the partials merged
//! by the box's last arrival, or by the split-keys merge launched after the partial routine.

use cubek_attention::kernels::{
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeShufflePartialAttentionRoutine,
};
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{DECODES, DECODES_WIDE_FP8, DECODES_WIDE_PER_TENSOR, Forward};

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

/// Every setting of the split-keys partial and of the last arrival over a decode step of every FP8
/// cache: the planes widen the `e4m3` keys or values into the sum as they read them.
#[test]
fn every_hypercube_shuffle_setting_reads_every_fp8_cache() {
    let partial = Forward::split_keys(HypercubeShufflePartialAttentionRoutine::new());
    partial.check(partial.at_every_setting(&DECODES_WIDE_FP8));
    let last_arrival = Forward::new(HypercubeLastArrivalShuffleAttentionRoutine::new());
    last_arrival.check(last_arrival.at_every_setting(&DECODES_WIDE_FP8));
}

/// Every setting of the split-keys partial and of the last arrival over a decode step of a cache
/// quantized per tensor: each cube's partial output leaves under the values' scale, its max and
/// sum under none.
#[test]
fn every_hypercube_shuffle_setting_reads_a_cache_quantized_per_tensor() {
    let partial = Forward::split_keys(HypercubeShufflePartialAttentionRoutine::new());
    partial.check(partial.at_every_setting(&DECODES_WIDE_PER_TENSOR));
    let last_arrival = Forward::new(HypercubeLastArrivalShuffleAttentionRoutine::new());
    last_arrival.check(last_arrival.at_every_setting(&DECODES_WIDE_PER_TENSOR));
}
