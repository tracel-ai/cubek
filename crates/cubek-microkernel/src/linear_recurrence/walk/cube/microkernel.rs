use super::settings::CubeWalkMicrokernelSettings;
use super::shape::HeldStateShape;
use crate::linear_recurrence::RecurrenceHeads;
use crate::linear_recurrence::walk::{RecurrenceWalkOperand, UpdateRule};
use cubek_std::microkernel::Microkernel;

/// A microkernel of the cube scope: one update rule walked over a [`HeldStateShape`], which the
/// cube routines place the same way whatever the rule.
pub trait CubeWalkMicrokernel:
    Copy
    + Microkernel<
        WorkItem = RecurrenceHeads,
        Settings = CubeWalkMicrokernelSettings,
        Operand = RecurrenceWalkOperand,
        CubeSpace = (),
    >
{
    /// The update rule it walks.
    const RULE: UpdateRule;

    /// How it holds a run's state.
    fn shape(&self) -> HeldStateShape;
}
