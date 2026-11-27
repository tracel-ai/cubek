//! The unit routine: each unit a box of one head's queries over every key it sees.

use cubek_attention::kernels::UnitAttentionRoutine;
use cubek_test_utils::ScopeChecker;

use crate::attention::{
    AttentionCase, DECODE, DECODES, Forward, PER_TENSOR, PROMPT, PROMPTS, SCORED_DECODES,
    SCORED_PAGED, SCORED_PROMPTS,
};

/// The unit routine's lead on every prompt and decode case.
#[test]
fn the_unit_routine_matches_the_host() {
    let forward = Forward::new(UnitAttentionRoutine);
    let cases = [PROMPTS.as_slice(), DECODES.as_slice()].concat();
    forward.check(forward.at_its_lead(&cases));
}

/// The unit routine's lead over an `f16` cache quantized per tensor, at a prompt and a decode
/// step: its register sums apply each side's scale as they read it.
#[test]
fn the_unit_routine_reads_a_cache_quantized_per_tensor() {
    let forward = Forward::new(UnitAttentionRoutine);
    let scaled = [PROMPT, DECODE].map(|case| AttentionCase {
        scales: Some(PER_TENSOR),
        ..case
    });
    forward.check(forward.at_its_lead(&scaled));
}

/// The unit routine's lead under both score terms, and each alone, on every prompt and decode
/// case, over a partly filled cache and over a paged one: the floor serves them on every device.
#[test]
fn the_unit_routine_takes_sinks_and_a_relative_bias() {
    let forward = Forward::new(UnitAttentionRoutine);
    let cases = [
        SCORED_PROMPTS.as_slice(),
        SCORED_DECODES.as_slice(),
        SCORED_PAGED.as_slice(),
    ]
    .concat();
    forward.check_some(forward.at_its_lead(&cases));
}
