//! The stream of plane fragments: every box's keys on one line its cubes share in runs, a run
//! holding a box whole writing it, and the runs sharing one leaving partials the last merges.

use cubek_attention::kernels::HypercubeStreamFragmentAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Forward, PROMPT_WIDE, PROMPTS, PROMPTS_WIDE_FP8, PROMPTS_WIDE_PER_TENSOR};

/// The routine's lead on every prompt case.
#[test]
fn the_hypercube_stream_routine_matches_the_host() {
    let forward = Forward::new(HypercubeStreamFragmentAttentionRoutine);
    forward.check(forward.at_its_lead(&PROMPTS));
}

/// Every setting the routine lists at a model's head dim.
#[test]
fn every_hypercube_stream_setting_matches_the_host_at_a_head_dim_of_128() {
    let forward = Forward::new(HypercubeStreamFragmentAttentionRoutine);
    forward.check(forward.at_every_setting(&[PROMPT_WIDE]));
}

/// Every setting the routine lists over every FP8 cache at a model's head dim: the score on the
/// device's FP8 instruction, the `f16` query converted ahead, and `e4m3` values widened to the
/// query's element by their stage.
#[test]
fn every_hypercube_stream_setting_reads_every_fp8_cache() {
    let forward = Forward::new(HypercubeStreamFragmentAttentionRoutine);
    forward.check(forward.at_every_setting(&PROMPTS_WIDE_FP8));
}

/// Every setting the routine lists over a cache quantized per tensor at a model's head dim: a run
/// holding a box whole and runs leaving partials alike drain under the values' scale.
#[test]
fn every_hypercube_stream_setting_reads_a_cache_quantized_per_tensor() {
    let forward = Forward::new(HypercubeStreamFragmentAttentionRoutine);
    forward.check(forward.at_every_setting(&PROMPTS_WIDE_PER_TENSOR));
}
