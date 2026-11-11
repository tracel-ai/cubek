//! The cube shuffle routine: a cube's planes of shuffles reading each step's keys where they lie.

use cubek_attention::kernels::CubeShuffleAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{DECODES, DECODES_WIDE_FP8, Forward, PROMPTS};

/// The cube shuffle routine's lead on every prompt and decode case.
#[test]
fn the_cube_shuffle_routine_matches_the_host() {
    let forward = Forward::new(CubeShuffleAttentionRoutine);
    let cases = [PROMPTS.as_slice(), DECODES.as_slice()].concat();
    forward.check(forward.at_its_lead(&cases));
}

/// Every setting over a decode step of every FP8 cache: the planes widen the `e4m3` keys or
/// values into the sum as they read them.
#[test]
fn every_cube_shuffle_setting_reads_every_fp8_cache() {
    let forward = Forward::new(CubeShuffleAttentionRoutine);
    forward.check(forward.at_every_setting(&DECODES_WIDE_FP8));
}
