//! The hypercubes of planes of shuffles: each box's keys cut across its cubes, the partials merged
//! by the box's last arrival, or by the split-keys merge launched after the partial routine.

use cubek_attention::kernels::{
    HypercubeLastArrivalShuffleAttentionRoutine, HypercubeShufflePartialAttentionRoutine,
};
use cubek_test_utils::ScopeChecker;

use crate::attention::{
    DECODE_WINDOWED, DECODES, DECODES_WIDE_FP8, DECODES_WIDE_PER_TENSOR, Forward, SCORED_DECODES,
    SCORED_PAGED,
};

/// Every setting of the split-keys partial and of the last arrival under both score terms on every
/// decode case, and their leads over a paged decode step: the cube whose run holds a row's first
/// key opens it at its sink, so each merge counts it once.
#[test]
fn every_hypercube_shuffle_setting_takes_sinks_and_a_relative_bias() {
    let partial = Forward::split_keys(HypercubeShufflePartialAttentionRoutine::new());
    partial.check(partial.at_every_setting(&SCORED_DECODES));
    partial.check(partial.at_its_lead(&SCORED_PAGED[1..]));
    let last_arrival = Forward::new(HypercubeLastArrivalShuffleAttentionRoutine::new());
    last_arrival.check(last_arrival.at_every_setting(&SCORED_DECODES));
    last_arrival.check(last_arrival.at_its_lead(&SCORED_PAGED[1..]));
}

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

/// Every setting of the split-keys partial and of the last arrival over a decode step that sees
/// only the last few of its keys: the boxes whose keys it sees none of hold nothing to merge, and
/// the merge must leave the boxes that do as they are.
#[test]
fn every_hypercube_shuffle_setting_decodes_through_a_window() {
    let partial = Forward::split_keys(HypercubeShufflePartialAttentionRoutine::new());
    partial.check(partial.at_every_setting(&[DECODE_WINDOWED]));
    let last_arrival = Forward::new(HypercubeLastArrivalShuffleAttentionRoutine::new());
    last_arrival.check(last_arrival.at_every_setting(&[DECODE_WINDOWED]));
}
