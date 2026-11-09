use super::settings::CubeWalkMicrokernelSettings;
use super::shape::HeldStateShape;
use crate::linear_recurrence::RecurrenceHeads;
use crate::linear_recurrence::walk::RecurrenceWalkOperand;
use cubek_std::microkernel::Microkernel;

/// A microkernel of the cube scope: a walk over a [`HeldStateShape`], which the cube routines
/// place the same way whatever the walk.
pub trait CubeWalkMicrokernel:
    Copy
    + Microkernel<
        WorkItem = RecurrenceHeads,
        Settings = CubeWalkMicrokernelSettings,
        Operand = RecurrenceWalkOperand,
        CubeSpace = (),
    >
{
    /// How it holds a run's state.
    fn shape(&self) -> HeldStateShape;
}
