use super::RecurrenceWalkOperand;
use serde::{Deserialize, Serialize};

/// How a position writes the state: what the walk adds along the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UpdateRule {
    /// `S = γS + k ⊗ β(v − (γS)ᵀk)`: the write is the delta from what the state recalls.
    Delta,
    /// `S = γS + k ⊗ βv`, read `Sᵀq + D v`: the write is added as it is.
    Additive,
}

impl UpdateRule {
    /// The operands the rule binds beyond those every walk binds.
    pub fn operands(self) -> &'static [RecurrenceWalkOperand] {
        match self {
            Self::Delta => &[],
            Self::Additive => &[RecurrenceWalkOperand::Skip],
        }
    }
}
