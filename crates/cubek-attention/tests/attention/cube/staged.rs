//! The cube staged routine: a cube's planes of fragments contracting the stages it fills.

use cubek_attention::kernels::CubeStagedAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{
    Forward, PROMPT, PROMPT_WIDE, PROMPTS, PROMPTS_WIDE_FP8, PROMPTS_WIDE_PER_TENSOR,
};

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

/// Every setting the routine lists over every FP8 cache at a model's head dim: the score on the
/// device's FP8 instruction, the `f16` query converted ahead, and `e4m3` values multiplied on it
/// beside the probabilities cast to `e4m3`, or widened to the query's element by their stage.
#[test]
fn every_cube_staged_setting_reads_every_fp8_cache() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_every_setting(&PROMPTS_WIDE_FP8));
}

/// Every setting the routine lists over a cache quantized per tensor at a model's head dim: the
/// stages hold the keys and values as stored, the softmax reads the score under the keys' scale,
/// and the output drains under the values'.
#[test]
fn every_cube_staged_setting_reads_a_cache_quantized_per_tensor() {
    let forward = Forward::new(CubeStagedAttentionRoutine);
    forward.check(forward.at_every_setting(&PROMPTS_WIDE_PER_TENSOR));
}
