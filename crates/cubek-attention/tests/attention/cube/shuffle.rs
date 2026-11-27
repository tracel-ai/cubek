//! The cube shuffle routine: a cube's planes of shuffles reading each step's keys where they lie.

use cubek_attention::kernels::CubeShuffleAttentionRoutine;
use cubek_test_utils::ScopeChecker;

use crate::attention::{
    DECODES, DECODES_WIDE_FP8, DECODES_WIDE_PER_TENSOR, Forward, PROMPTS, PROMPTS_WIDE_FP8,
    PROMPTS_WIDE_PER_TENSOR, SCORED_DECODES, SCORED_PAGED, SCORED_PROMPTS,
};

/// The cube shuffle routine's lead under both score terms, and each alone, on every prompt and
/// decode case and over a paged cache: its planes add the relative bias to each score they hold
/// and open each row at its sink.
#[test]
fn the_cube_shuffle_routine_takes_sinks_and_a_relative_bias() {
    let forward = Forward::new(CubeShuffleAttentionRoutine);
    let cases = [
        SCORED_PROMPTS.as_slice(),
        SCORED_DECODES.as_slice(),
        SCORED_PAGED.as_slice(),
    ]
    .concat();
    forward.check(forward.at_its_lead(&cases));
}

/// The cube shuffle routine's lead on every prompt and decode case.
#[test]
fn the_cube_shuffle_routine_matches_the_host() {
    let forward = Forward::new(CubeShuffleAttentionRoutine);
    let cases = [PROMPTS.as_slice(), DECODES.as_slice()].concat();
    forward.check(forward.at_its_lead(&cases));
}

/// Every setting over a decode step of every FP8 cache, and the lead over a prompt of each: the
/// planes widen the `e4m3` keys or values into the sum as they read them, the route a prompt
/// over `e4m3` values takes.
#[test]
fn every_cube_shuffle_setting_reads_every_fp8_cache() {
    let forward = Forward::new(CubeShuffleAttentionRoutine);
    forward.check(forward.at_every_setting(&DECODES_WIDE_FP8));
    forward.check(forward.at_its_lead(&PROMPTS_WIDE_FP8));
}

/// Every setting over a decode step of a cache quantized per tensor, and the lead over a prompt of
/// it: the planes read the keys and values as stored, and take each side's scale on once, the
/// keys' on every score and the values' on every weight.
#[test]
fn every_cube_shuffle_setting_reads_a_cache_quantized_per_tensor() {
    let forward = Forward::new(CubeShuffleAttentionRoutine);
    forward.check(forward.at_every_setting(&DECODES_WIDE_PER_TENSOR));
    forward.check(forward.at_its_lead(&PROMPTS_WIDE_PER_TENSOR));
}
