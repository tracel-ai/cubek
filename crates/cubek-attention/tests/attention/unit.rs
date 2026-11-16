//! The unit routine: each unit a box of one head's queries over every key it sees.

use cubek_attention::kernels::UnitAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{AttentionCase, DECODE, DECODES, Forward, PER_TENSOR, PROMPT, PROMPTS};

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
