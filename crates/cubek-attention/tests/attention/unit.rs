//! The unit routine: each unit a box of one head's queries over every key it sees.

use cubek_attention::kernels::UnitAttentionRoutine;
use cubek_test_utils::ScopeUnderTest;

use crate::attention::{DECODES, Forward, PROMPTS};

/// The unit routine's lead on every prompt and decode case.
#[test]
fn the_unit_routine_matches_the_host() {
    let forward = Forward::new(UnitAttentionRoutine);
    let cases = [PROMPTS.as_slice(), DECODES.as_slice()].concat();
    forward.check(forward.at_its_lead(&cases));
}
