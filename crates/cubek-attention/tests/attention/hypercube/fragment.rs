//! The hypercube of plane fragments: each box's keys cut across its cubes, a second launch merging
//! the partials they leave.

use cubek_attention::kernels::HypercubeSecondLaunchFragmentAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{Forward, PROMPT_WIDE, PROMPTS, PROMPTS_WIDE_FP8};

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

/// Every microkernel setting the routine lists, on eight cubes a box, over `e4m3` keys at a
/// model's head dim: multiplied on the device's FP8 instruction, the query converted as it is
/// staged, or widened to the query's `f16`. The cubes a box splits over change the merge, not the
/// score, so one count stands for them.
#[test]
fn every_hypercube_fragment_microkernel_reads_fp8_keys() {
    let forward = Forward::new(HypercubeSecondLaunchFragmentAttentionRoutine);
    let eight_cubes = forward
        .settings()
        .into_iter()
        .filter(|settings| settings.cubes_along_keys.get() == 8);
    let [fp8_keys, ..] = PROMPTS_WIDE_FP8;
    forward.check(forward.at(eight_cubes, &[fp8_keys]));
}

/// The routine's first microkernel settings, on eight cubes a box, over every FP8 cache: `e4m3`
/// values are widened to the query's `f16` as they are staged, beside `f16` keys or `e4m3` ones.
/// The lead, a cmma fragment that reads its values through windows, declines values it would
/// widen; the manual `mma` fragments after it read them.
#[test]
fn the_hypercube_fragment_routine_reads_every_fp8_cache() {
    let forward = Forward::new(HypercubeSecondLaunchFragmentAttentionRoutine);
    let first = forward
        .settings()
        .into_iter()
        .filter(|settings| settings.cubes_along_keys.get() == 8)
        .take(3);
    forward.check(forward.at(first, &PROMPTS_WIDE_FP8));
}
