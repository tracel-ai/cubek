//! The hypercube of plane fragments: each box's keys cut across its cubes, the split-keys merge
//! launched after it merging the partials they leave.

use cubek_attention::kernels::HypercubeFragmentPartialAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Forward, PROMPT_WIDE, PROMPTS, PROMPTS_WIDE_FP8, PROMPTS_WIDE_PER_TENSOR};

/// The routine's lead, merged, on every prompt case.
#[test]
fn the_hypercube_fragment_routine_matches_the_host() {
    let forward = Forward::split_keys(HypercubeFragmentPartialAttentionRoutine);
    forward.check(forward.at_its_lead(&PROMPTS));
}

/// Every setting the routine lists at a model's head dim: the stages each cube hands on to the
/// partial it leaves are each a different kernel.
#[test]
fn every_hypercube_fragment_setting_matches_the_host_at_a_head_dim_of_128() {
    let forward = Forward::split_keys(HypercubeFragmentPartialAttentionRoutine);
    forward.check(forward.at_every_setting(&[PROMPT_WIDE]));
}

/// Every microkernel setting the routine lists, on eight cubes a box, over every FP8 cache at a
/// model's head dim: `e4m3` keys multiplied on the device's FP8 instruction, the query converted
/// to `e4m3` by its stage, or widened to the query's `f16` by theirs; `e4m3` values widened to it
/// by theirs.
/// The cubes a box splits over change the merge, not the score, so one count stands for them.
#[test]
fn every_hypercube_fragment_microkernel_reads_every_fp8_cache() {
    let forward = Forward::split_keys(HypercubeFragmentPartialAttentionRoutine);
    let eight_cubes = forward
        .settings()
        .into_iter()
        .filter(|settings| settings.cubes_along_keys.get() == 8);
    forward.check(forward.at(eight_cubes, &PROMPTS_WIDE_FP8));
}

/// Every microkernel setting the routine lists, on eight cubes a box, over a cache quantized per
/// tensor at a model's head dim: each cube's partial output leaves under the values' scale, its
/// max and sum under none, so the merge reads them as any other.
#[test]
fn every_hypercube_fragment_microkernel_reads_a_cache_quantized_per_tensor() {
    let forward = Forward::split_keys(HypercubeFragmentPartialAttentionRoutine);
    let eight_cubes = forward
        .settings()
        .into_iter()
        .filter(|settings| settings.cubes_along_keys.get() == 8);
    forward.check(forward.at(eight_cubes, &PROMPTS_WIDE_PER_TENSOR));
}
