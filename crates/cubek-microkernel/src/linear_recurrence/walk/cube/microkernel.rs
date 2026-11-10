use super::settings::CubeWalkMicrokernelSettings;
use super::shape::HeldStateShape;
use crate::linear_recurrence::walk::{RecurrenceWalkOperand, UpdateRule};
use crate::linear_recurrence::{
    HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION, RATE, RecurrenceHeads, VALUE_FEATURE,
};
use cubecl::tile::{Axis, Levels};
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

    /// The levels inside one run's units, which a routine states the rest of the partitioning
    /// around: a unit's leaf of its rows and columns, walked a position at a time, and the units
    /// cutting the key rows of a run. `kept` is the axis the kept states are cut along, where the
    /// states are owned.
    fn tiling(&self, kept: Option<(Axis, usize)>) -> Levels {
        let shape = self.shape();
        let mut axes = vec![
            (POSITION, 1),
            (RATE, 2),
            (KEY_HEAD, 1),
            (HEAD_IN_GROUP, 1),
            (KEY_FEATURE, shape.rows()),
            (VALUE_FEATURE, shape.columns()),
        ];
        axes.extend(kept);
        Levels::leaf(&axes)
            .walk_every(&[POSITION])
            .units(&[(KEY_FEATURE, shape.units())])
    }
}
