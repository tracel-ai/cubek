use super::settings::CubeWalkMicrokernelSettings;
use super::shape::HeldStateShape;
use crate::linear_recurrence::walk::{RecurrenceWalkItem, RecurrenceWalkOperand, UpdateRule};
use crate::linear_recurrence::{
    HEAD_IN_GROUP, KEY_FEATURE, KEY_HEAD, POSITION, RATE, VALUE_FEATURE,
};
use cubecl::tile::launch::Vectorization;
use cubecl::tile::{Axis, Levels};
use cubek_std::Hardware;
use cubek_std::microkernel::{CubePlanes, KernelEstimate, Microkernel};

/// A cube's runs of a head's value columns walked through every position one at a time under an
/// [`UpdateRule`], held as its [`HeldStateShape`] states. Every rule is placed the same way, so
/// the rule is a field and not a type.
///
/// It reads no plane instruction and no plane width, only a cube of units and shared memory to
/// meet in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeWalkMicrokernel {
    pub shape: HeldStateShape,
    pub rule: UpdateRule,
}

impl CubeWalkMicrokernel {
    /// The levels inside one run's units, which a routine states the rest of the partitioning
    /// around: a unit's leaf of its rows and columns, walked a position at a time, and the units
    /// cutting the key rows of a run. `kept` is the axis the kept states are cut along, where the
    /// states are owned.
    pub fn tiling(&self, kept: Option<(Axis, usize)>) -> Levels {
        let mut axes = vec![
            (POSITION, 1),
            (RATE, 2),
            (KEY_HEAD, 1),
            (HEAD_IN_GROUP, 1),
            (KEY_FEATURE, self.shape.rows()),
            (VALUE_FEATURE, self.shape.columns()),
        ];
        axes.extend(kept);
        Levels::leaf(&axes)
            .walk_every(&[POSITION])
            .units(&[(KEY_FEATURE, self.shape.units())])
    }
}

impl core::fmt::Display for CubeWalkMicrokernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.shape.fmt(f)
    }
}

impl Microkernel for CubeWalkMicrokernel {
    type WorkItem = RecurrenceWalkItem;
    type Settings = CubeWalkMicrokernelSettings;
    type Operand = RecurrenceWalkOperand;
    type CubeSpace = ();

    const DECLINE: &'static str = HeldStateShape::DECLINE;

    fn settings(_hw: &Hardware) -> Vec<Self::Settings> {
        CubeWalkMicrokernelSettings::listed()
    }

    fn candidates(
        hw: &Hardware,
        item: &RecurrenceWalkItem,
        settings: &Self::Settings,
    ) -> impl Iterator<Item = Self> {
        HeldStateShape::new(hw, &item.heads, settings)
            .map(|shape| Self {
                shape,
                rule: item.rule,
            })
            .into_iter()
    }

    /// Every operand as [`RecurrenceWalkOperand::require`] states it, a run `columns` wide and a
    /// unit's rows.
    fn vectorize(&self, _: &(), vectorization: &mut Vectorization<RecurrenceWalkOperand>) {
        RecurrenceWalkOperand::require(
            self.shape.columns(),
            self.shape.rows(),
            self.rule,
            vectorization,
        );
    }

    fn estimate(
        &self,
        _item: &RecurrenceWalkItem,
        _: &(),
        _planes: CubePlanes,
        estimate: &mut KernelEstimate,
    ) {
        self.shape.estimate(estimate);
    }
}
