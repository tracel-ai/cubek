use super::UpdateRule;
use crate::linear_recurrence::RecurrenceHeads;

/// What a walk microkernel is chosen for: the heads and the rule their positions are walked under.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RecurrenceWalkItem {
    pub heads: RecurrenceHeads,
    pub rule: UpdateRule,
}
